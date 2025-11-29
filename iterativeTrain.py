"""
AlphaZero-style self-play training for your 'ChessModel'
Run on your LOCAL GPU
(MODIFIED to use a 'teacher' model for Iteration 0)
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import os
import shutil
import torch.multiprocessing as mp
from functools import partial

# --- 1. MODEL DEFINITION (Unchanged) ---

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):t
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessModel(nn.Module):
    def __init__(self, in_planes=13, blocks=5, channels=128):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.layers = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.conv_policy = nn.Conv2d(channels, 64, kernel_size=1) 
        self.bn_policy = nn.BatchNorm2d(64)
        self.conv_value = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8*1, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layers(x)
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(-1, 4096)
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(-1, 8*8)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        return p, v

# --- 2. HELPER FUNCTIONS (Unchanged) ---

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros(13, 8, 8)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            row, col = divmod(sq, 8)
            plane = piece_map[piece.symbol()]
            tensor[plane, 7 - row, col] = 1
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1
    return tensor

def move_to_index(move):
    return move.from_square * 64 + move.to_square

class MCTSNode:
    # (This class is unchanged)
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
    def select_child(self, c_puct=1.0):
        import math
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            q_value = child.value()
            u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child
    def expand(self, policy_probs):
        for move, prob in policy_probs.items():
            if move not in self.children:
                next_board = self.board.copy()
                next_board.push(move)
                self.children[move] = MCTSNode(next_board, parent=self, move=move, prior=prob)
        self.is_expanded = True
    def backup(self, value):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

# --- 3. CORE SELF-PLAY & TRAINING (Unchanged) ---

def mcts_search(model, board, device, num_simulations=100):
    # (This function is unchanged)
    root = MCTSNode(board)
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        while node.is_expanded and not node.board.is_game_over():
            node = node.select_child()
            search_path.append(node)
        value = 0.0
        if node.board.is_game_over():
            result = node.board.result()
            if result == "1-0": value = 1.0 if node.board.turn == chess.WHITE else -1.0
            elif result == "0-1": value = -1.0 if node.board.turn == chess.WHITE else 1.0
        else:
            with torch.no_grad():
                board_tensor = board_to_tensor(node.board).unsqueeze(0).to(device) 
                policy_logits, value_pred = model(board_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                value = value_pred.item()
            legal_moves = list(node.board.legal_moves)
            move_probs = {}
            total_prob = 0.0
            for move in legal_moves:
                idx = move_to_index(move)
                if 0 <= idx < 4096:
                    move_probs[move] = policy_probs[idx]
                    total_prob += policy_probs[idx]
            if total_prob > 0: move_probs = {m: p / total_prob for m, p in move_probs.items()}
            elif legal_moves: move_probs = {m: 1.0 / len(legal_moves) for m in legal_moves}
            node.expand(move_probs)
        for n in reversed(search_path):
            n.backup(value)
            value = -value
    moves = []
    visits = []
    for move, child in root.children.items():
        moves.append(move)
        visits.append(child.visit_count)
    if not visits: return {}
    visits = np.array(visits, dtype=np.float32)
    probs = visits / visits.sum()
    return {move: prob for move, prob in zip(moves, probs)}

def play_self_play_game(model, device, num_simulations):
    # (This function is unchanged)
    board = chess.Board()
    training_data = []
    move_count = 0
    while not board.is_game_over() and move_count < 200:
        action_probs = mcts_search(model, board, device, num_simulations=num_simulations)
        if not action_probs: break
        state = board_to_tensor(board).numpy()
        training_data.append((state, action_probs.copy(), None))
        moves = list(action_probs.keys())
        probs = np.array(list(action_probs.values()), dtype=np.float64)
        probs = probs / probs.sum()
        move = np.random.choice(moves, p=probs)
        board.push(move)
        move_count += 1
    result = board.result()
    if result == "1-0": winner_value = 1.0
    elif result == "0-1": winner_value = -1.0
    else: winner_value = 0.0
    final_data = []
    for i, (state, probs, _) in enumerate(training_data):
        outcome = winner_value if (i % 2) == 0 else -winner_value
        final_data.append((state, probs, outcome))
    return final_data, result

class SelfPlayDataset(Dataset):
    # (This class is unchanged)
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        state, move_probs, outcome = self.data[idx]
        policy_target = np.zeros(4096, dtype=np.float32)
        for move, prob in move_probs.items():
            policy_target[move_to_index(move)] = prob
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(policy_target),
            torch.FloatTensor([outcome])
        )

# --- 4. NEW MULTIPROCESSING WORKER (Unchanged) ---

def self_play_worker(game_num, num_simulations, model_state_dict, device_str):
    # (This function is unchanged)
    print(f"[Worker {game_num}]: Process started.")
    device = torch.device(device_str)
    model = ChessModel().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    final_data, result = play_self_play_game(model, device, num_simulations)
    print(f"[Worker {game_num}]: Game finished. Result={result} ({len(final_data)} positions)")
    return final_data, result

# --- 5. MAIN ITERATION FUNCTION (Modified) ---

# --- ADDED 'teacher_model_path' ARGUMENT ---
def self_play_iteration(iteration: int, device, games_per_iteration: int, num_simulations: int, model_dir: str, teacher_model_path: str):
    print(f"\n{'='*60}")
    print(f"SELF-PLAY ITERATION {iteration}")
    print(f"{'='*60}\n")

    model = ChessModel().to(device)

    # --- THIS IS THE NEW LOADING LOGIC ---
    if iteration == 0:
        # On the first iteration, try to load the 'teacher' model
        if os.path.exists(teacher_model_path):
            print(f"--- WARM START ---")
            print(f"Loading 'teacher' model from {teacher_model_path} for Iteration 0.")
            model.load_state_dict(torch.load(teacher_model_path, map_location=device))
        else:
            print(f"--- COLD START ---")
            print(f"No teacher model found at {teacher_model_path}. Starting Iteration 0 from scratch.")
    else:
        # On all other iterations, load the previous checkpoint
        model_path = os.path.join(model_dir, f'chess_model_iter{iteration-1}.pth')
        if os.path.exists(model_path):
            print(f"Loading model from {model_path} for Iteration {iteration}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"--- ERROR ---")
            print(f"Warning: Could not find checkpoint {model_path}. Starting Iteration {iteration} from scratch.")
    # --- END NEW LOADING LOGIC ---

    model.eval()

    # (The multiprocessing logic is unchanged)
    print(f"\nGenerating {games_per_iteration} self-play games using 4 processes...")
    model.cpu()
    model_state_dict = model.state_dict()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    worker_partial = partial(self_play_worker, 
                             num_simulations=num_simulations, 
                             model_state_dict=model_state_dict, 
                             device_str=device_str)
    with mp.Pool(4) as pool:
        results = pool.map(worker_partial, range(games_per_iteration))
    model.to(device)
    all_training_data = []
    for game_data, result in results:
        all_training_data.extend(game_data)

    print(f"\nTotal training positions: {len(all_training_data)}")
    if not all_training_data:
        print("No training data generated, skipping iteration.")
        return

    # (The training loop is unchanged)
    dataset = SelfPlayDataset(all_training_data)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"\nTraining on self-play data...")
    model.train() 
    value_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    for epoch in range(epochs):
        total_p, total_v, num = 0, 0, 0
        for boards, policy_targets, value_targets in dataloader:
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            optimizer.zero_grad()
            policy_pred, value_pred = model(boards)
            p_loss = -(policy_targets * torch.log_softmax(policy_pred, dim=1)).sum(dim=1).mean()
            v_loss = value_criterion(value_pred.squeeze(), value_targets.squeeze())
            loss = p_loss + v_loss
            loss.backward()
            optimizer.step()
            total_p += p_loss.item()
            total_v += v_loss.item()
            num += 1
        avg_p = total_p / (num + 1e-9)
        avg_v = total_v / (num + 1e-9)
        print(f"Epoch {epoch+1}/{epochs}: Policy Loss={avg_p:.4f}, Value Loss={avg_v:.4f}")

    # (The model saving logic is unchanged)
    model_cpu = model.cpu()
    save_path = os.path.join(model_dir, f'chess_model_iter{iteration}.pth')
    torch.save(model_cpu.state_dict(), save_path)
    
    print(f"\nModel saved to {save_path}")
    print(f"Iteration {iteration} complete!")
    return save_path

# --- 6. MAIN EXECUTION BLOCK (Modified) ---

if __name__ == "__main__":
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # --- Configuration ---
    num_iterations = 15
    games_per_iteration = 100
    num_simulations = 1000
    
    # --- Local Setup ---
    MODEL_DIR = "./chess_models"
    
    # --- THIS IS YOUR 'TEACHER' MODEL PATH ---
    BOT_MODEL_PATH = "./model.pth" 
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(BOT_MODEL_PATH), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using Main Process GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No GPU found, using Main Process CPU")

    print(f"\n{'='*60}")
    print(f"LOCAL ALPHAZERO SELF-PLAY TRAINING (for ChessModel)")
    print(f"  Teacher Model Path: {BOT_MODEL_PATH}") # Added for clarity
    print(f"{'='*60}")
    # ... (rest of prints are the same) ...
    print(f"Iterations: {num_iterations}")
    print(f"Games per iteration: {games_per_iteration}")
    print(f"MCTS simulations: {num_simulations}")
    print(f"Models will be saved to: {MODEL_DIR}")
    print(f"{'='*60}\n")

    latest_model_path = ""
    for iteration in range(num_iterations):
        
        # --- PASS THE 'TEACHER' PATH TO THE FUNCTION ---
        latest_model_path = self_play_iteration(
            iteration,
            device=device,
            games_per_iteration=games_per_iteration,
            num_simulations=num_simulations,
            model_dir=MODEL_DIR,
            teacher_model_path=BOT_MODEL_PATH  # <-- NEW
        )

    if latest_model_path:
        print(f"\n{'='*60}")
        print(f"SELF-PLAY TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Final model: {latest_model_path}")

        # This will now OVERWRITE your original 'teacher' model.pth
        # with the new, fine-tuned, and stronger version.
        shutil.copy(latest_model_path, BOT_MODEL_PATH)
        print(f"✓ Automatically deployed to {BOT_MODEL_PATH}")
    else:
        print("Training finished, but no model was saved.")
