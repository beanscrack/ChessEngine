# pipeline.py
import os
import io
import zstandard as zstd
import multiprocessing
from contextlib import closing
import torch
import chess
import rust_pgn_reader_python_binding as pgn_reader
import codecs

# ---------- Config ----------
PGN_ZST_PATH = "lichess_oct.pgn.zst"
OUTPUT_DIR = "training_shards"
POSITIONS_PER_SHARD = 100_000_00         # shard size = number of positions
NUM_WORKERS = max(1, multiprocessing.cpu_count())
CHUNKSIZE = 64                           # imap_unordered chunksize
# ----------------------------

def move_to_index(move_uci: str) -> int:
    """Map UCI move to integer index. Keep this consistent with your model."""
    try:
        move = chess.Move.from_uci(move_uci)
        return move.from_square * 64 + move.to_square
    except Exception:
        return -1

def move_extractor_to_serializable(game_obj):
    """
    Convert MoveExtractor (Rust object) into a small Python dict that is picklable.
    Structure:
      {
        "headers": { ... },
        "moves": ["e2e4", "e7e5", ...],
        "outcome": "1-0" or "0-1" or "1/2-1/2" or "*"
      }
    """
    # headers: try to turn into a dict
    headers = None
    try:
        # If it's a mapping-like object this will turn it into a dict
        headers = dict(game_obj.headers)
    except Exception:
        # fallback: try attribute access or leave as empty dict
        try:
            headers = game_obj.headers if game_obj.headers is not None else {}
        except Exception:
            headers = {}

    # moves: safe extraction
    moves = []
    try:
        for m in game_obj.moves:
            # Try common attributes/methods used by bindings
            uci = None
            if hasattr(m, "uci"):
                u = getattr(m, "uci")
                uci = u() if callable(u) else u
            elif hasattr(m, "to_uci"):
                t = getattr(m, "to_uci")
                uci = t() if callable(t) else t
            else:
                # last-resort: str(m)
                uci = str(m)
            if uci is None:
                continue
            moves.append(uci)
    except Exception:
        # if moves attribute isn't iterable, try another access (defensive)
        moves = []

    outcome = None
    try:
        outcome = getattr(game_obj, "outcome", None)
        # if outcome is not a simple string, try to use headers
        if not outcome:
            outcome = headers.get("Result", "*")
    except Exception:
        outcome = headers.get("Result", "*")

    return {"headers": headers, "moves": moves, "outcome": outcome}

def process_parsed_game(game_dict):
    """
    Convert parsed dict -> training tuples.
    Returns list of (fen_before_move, policy_index, value_target)
    """
    headers = game_dict.get("headers", {})
    moves = game_dict.get("moves", [])
    result = headers.get("Result", game_dict.get("outcome", "*"))

    if result == "1-0":
        value_target = [1.0, 0.0, 0.0]
    elif result == "0-1":
        value_target = [0.0, 0.0, 1.0]
    elif result == "1/2-1/2":
        value_target = [0.0, 1.0, 0.0]
    else:
        # skip games without final result
        return None

    training_tuples = []
    board = chess.Board()

    # custom starting FEN
    start_fen = headers.get("FEN")
    if start_fen:
        try:
            board.set_fen(start_fen)
        except Exception:
            return None

    for move_uci in moves:
        if not move_uci:
            continue
        fen_before = board.fen()
        policy_index = move_to_index(move_uci)
        if policy_index == -1:
            # illegal/malformed move; skip this move (but continue)
            try:
                board.push_uci(move_uci)
            except Exception:
                break
            continue

        training_tuples.append((fen_before, policy_index, value_target))
        try:
            board.push_uci(move_uci)
        except Exception:
            # if pushing fails, stop processing this game
            break

    return training_tuples if training_tuples else None

# Multiprocessing worker wrapper expects a simple picklable object (dict)
def worker_process(game_dict):
    try:
        return process_parsed_game(game_dict)
    except Exception:
        return None

# ---------- streaming helpers ----------
def stream_pgn_games_from_zst(zst_path):
    """
    Read a .zst file and yield raw PGN strings one at a time.
    Uses an incremental UTF-8 decoder with errors='replace' to avoid truncation.
    Detects game boundaries when a line starts with '[Event ' (typical PGN).
    """
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as fh:
        reader = dctx.stream_reader(fh)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buf_lines = []
        partial = b""

        # read in 1MB chunks (tune if needed)
        CHUNK_SIZE = 2**20
        while True:
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                break
            text = decoder.decode(chunk)
            # iterate lines
            parts = text.splitlines(keepends=True)
            for line in parts:
                # normalize line start
                if line.startswith("[Event "):
                    # start of a new game -> yield previous (if any)
                    if buf_lines:
                        yield "".join(buf_lines)
                        buf_lines = []
                buf_lines.append(line)

        # flush any buffered text
        tail = decoder.decode(b"", final=True)
        if tail:
            for line in tail.splitlines(keepends=True):
                if line.startswith("[Event "):
                    if buf_lines:
                        yield "".join(buf_lines)
                        buf_lines = []
                buf_lines.append(line)

        if buf_lines:
            yield "".join(buf_lines)
# ---------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Starting pipeline with {NUM_WORKERS} workers...")
    total_games = 0
    total_positions = 0
    shard_index = 0
    current_shard_data = []

    # IMPORTANT: do not pass non-picklable Rust objects to workers.
    # We'll convert MoveExtractor -> dict in the main process, then send dicts to workers.
    with closing(multiprocessing.Pool(processes=NUM_WORKERS)) as pool:
        try:
            # 1) stream raw PGN strings
            raw_game_iter = stream_pgn_games_from_zst(PGN_ZST_PATH)

            # 2) convert to MoveExtractor (in main) then to plain dict
            def parsed_game_generator():
                for raw_pgn in raw_game_iter:
                    try:
                        me = pgn_reader.parse_game(raw_pgn)
                        game_dict = move_extractor_to_serializable(me)
                        yield game_dict
                    except Exception:
                        # skip malformed games
                        continue

            # 3) parallel processing: pass plain dicts into worker_process
            for game_tuples in pool.imap_unordered(worker_process, parsed_game_generator(), chunksize=CHUNKSIZE):
                total_games += 1
                if not game_tuples:
                    continue

                current_shard_data.extend(game_tuples)

                if len(current_shard_data) >= POSITIONS_PER_SHARD:
                    shard_file = os.path.join(OUTPUT_DIR, f"shard_{shard_index:04d}.pt")
                    torch.save(current_shard_data, shard_file)
                    print(f"Saved {shard_file} ({len(current_shard_data)} positions). Total games: {total_games}")
                    total_positions += len(current_shard_data)
                    shard_index += 1
                    current_shard_data = []

        except KeyboardInterrupt:
            print("\nCaught interrupt, shutting down workers...")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if current_shard_data:
                shard_file = os.path.join(OUTPUT_DIR, f"shard_{shard_index:04d}.pt")
                torch.save(current_shard_data, shard_file)
                total_positions += len(current_shard_data)
                print(f"Saved final {shard_file} ({len(current_shard_data)} positions).")

            print("\nPipeline Complete.")
            print(f"Total games processed: {total_games}")
            print(f"Total positions processed: {total_positions}")
            print(f"Total shards created: {shard_index + (1 if current_shard_data else 0)}")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
