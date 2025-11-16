# --- Imports: Added GameContext and chess_manager ---
from .utils import chess_manager, GameContext
import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.pgn
from chess import Move
import random

# ---------- 1. MODEL DEFINITION ----------
# (This section is identical to your code and correct)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
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
        
        # --- THIS IS THE FIX ---
        self.conv_policy = nn.Conv2d(channels, 64, kernel_size=1) # Was Conv2t
        # --- END FIX ---
        
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

# ---------- 2. HELPER FUNCTIONS ----------
# (This section is identical to your code and correct)

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

def move_to_index(move: chess.Move) -> int:
    """Map a chess.Move object to an integer index (0-4095)."""
    return move.from_square * 64 + move.to_square

# ---------- 3. MODEL LOADING ----------
# (This section is identical to your code and correct)

print("Loading model...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Use try/except for safer loading ---
MODEL = None
try:
    script_dir = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(script_dir, "weights", "model.pth")
    MODEL = ChessModel().to(DEVICE)
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    MODEL.eval() # Set model to evaluation mode
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please make sure 'model.pth' is in the 'src/weights/' directory.")
except Exception as e:
    print(f"Error loading model: {e}")

# ---------- 4. BOT ENTRYPOINT ----------
# This is the new function that replaces get_move
# It uses the decorator from utils.py

@chess_manager.entrypoint
def find_best_move(ctx: GameContext) -> Move:
    """
    This function is called by the framework.
    It receives a GameContext and must return a chess.Move object.
    """
    
    if MODEL is None:
        print("Model not loaded. Returning null move.")
        return Move.null()

    # 1. Get the board from the context (no PGN parsing needed!)
    board = ctx.board
    
    # 2. Get all legal moves
    legal_moves = list(board.generate_legal_moves())
    
    if not legal_moves:
        print("No legal moves available. Returning null move.")
        return Move.null() # Return a null move object

    # 3. Convert board to tensor
    input_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)

    # 4. Get model prediction
    with torch.no_grad():
        policy_logits, value_pred = MODEL(input_tensor)

    # 5. Convert logits to probabilities
    all_move_probs = F.softmax(policy_logits, dim=1)[0]
    
    # 6. Filter: Get the model's probability *only for legal moves*
    legal_move_probs = {}
    total_legal_prob = 0.0
    
    for move in legal_moves:
        idx = move_to_index(move)
        
        # This is your correct 4096 check
        if 0 <= idx < 4096:
            prob = all_move_probs[idx].item()
            legal_move_probs[move] = prob
            total_legal_prob += prob

    # 7. Renormalize probabilities 
    final_probs = {}
    if total_legal_prob == 0.0:
        print("Warning: Model assigned 0 prob to all legal moves. Picking random.")
        # Assign uniform probability if model fails
        for move in legal_moves:
            final_probs[move] = 1.0 / len(legal_moves)
        best_move = random.choice(legal_moves)
    else:
        # Renormalize
        final_probs = {
            move: prob / total_legal_prob
            for move, prob in legal_move_probs.items()
        }
        # 8. Choose the best move
        best_move = max(final_probs, key=final_probs.get)

    # 9. Log probabilities back to the UI (this is a cool feature of the framework)
    ctx.logProbabilities(final_probs)
    
    # 10. Return the chess.Move object (NOT a string)
    print(f"Model chose: {best_move.uci()} (Value: {value_pred.item():.4f})")
    return best_move