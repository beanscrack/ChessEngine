import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader
import glob
import chess
import os
import random
import argparse
import math

# --- OPTIMIZATION: Enable TF32 & BF16 Support ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- 1. MODEL DEFINITION ---
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
    def __init__(self, in_planes=18, blocks=10, channels=256):
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

# --- 2. TENSORIZATION ---
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros(18, 8, 8)
    piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            row, col = divmod(sq, 8)
            tensor[piece_map[piece.symbol()], 7-row, col] = 1
    if board.turn == chess.WHITE: tensor[12, :, :] = 1
    if board.has_kingside_castling_rights(chess.WHITE): tensor[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE): tensor[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK): tensor[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK): tensor[16, :, :] = 1
    if board.ep_square:
        row, col = divmod(board.ep_square, 8)
        tensor[17, 7-row, col] = 1
    return tensor

# --- 3. DATASET ---
class ShardIterableDataset(IterableDataset):
    def __init__(self, shard_directory):
        super().__init__()
        self.shard_directory = shard_directory
        self.shard_files = glob.glob(os.path.join(shard_directory, "shard_*.pt"))
        print(f"Found {len(self.shard_files)} shards.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_to_process = self.shard_files.copy()
        random.shuffle(files_to_process)

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_for_this_worker = files_to_process[worker_id::num_workers]
        else:
            files_for_this_worker = files_to_process

        for shard_file in files_for_this_worker:
            try:
                shard_data = torch.load(shard_file)
                random.shuffle(shard_data)
                
                for fen, policy_idx, value_target in shard_data:
                    board = chess.Board(fen)
                    final_val = 0.0
                    if hasattr(value_target, '__len__') and len(value_target) == 3:
                        if value_target[0] == 1: 
                            final_val = 1.0 
                        elif value_target[2] == 1: 
                            final_val = -1.0 
                        else: 
                            final_val = 0.0 
                    else:
                        final_val = float(value_target)
                    
                    yield board_to_tensor(board), policy_idx, torch.tensor([final_val], dtype=torch.float32)

            except Exception as e:
                print(f"Warning: Skipping corrupted shard {shard_file}: {e}")
                continue

# --- CHECKPOINT UTILS ---
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer, scheduler):
    print(f"=> Loading checkpoint from {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['batch_idx'], checkpoint['epoch']

# --- 4. TRAINING SCRIPT ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ChessModel().to(device) 
    
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    except Exception as e:
        print(f"Could not compile: {e}")

    # --- BF16 CONFIGURATION ---
    BATCH_SIZE = 32_768 
    

    MAX_LR = 6e-3 
    
    NUM_EPOCHS = 3

    base_steps = 5_927_764
    scaling_factor = BATCH_SIZE / 1024.0
    
    BATCHES_PER_EPOCH = int(base_steps / scaling_factor)
    TOTAL_BATCHES = BATCHES_PER_EPOCH * NUM_EPOCHS 
    
    print(f"--- CONFIGURATION ---")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Scaled LR:  {MAX_LR} (BFloat16 Mode)")
    print(f"Steps/Epoch: {BATCHES_PER_EPOCH} (Total: {TOTAL_BATCHES})")

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=TOTAL_BATCHES,
        pct_start=0.06, anneal_strategy='cos'
    )
    
    # NOTE: NO SCALER NEEDED FOR BFLOAT16

    start_batch_idx = 0
    start_epoch = 0
    
    if args.resume and os.path.exists(args.resume):
        start_batch_idx, start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
    elif os.path.exists("latest_checkpoint.pth.tar"):
         print("Found latest_checkpoint.pth.tar, resuming...")
         start_batch_idx, start_epoch = load_checkpoint("latest_checkpoint.pth.tar", model, optimizer, scheduler)

    policy_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    value_loss_fn = nn.MSELoss()

    data_dir = os.path.expanduser(args.data_path)
    dataset = ShardIterableDataset(shard_directory=data_dir)
    
    NUM_WORKERS = 32 
    
    loader = DataLoader(dataset, 
                        batch_size=BATCH_SIZE, 
                        num_workers=NUM_WORKERS, 
                        pin_memory=True,
                        persistent_workers=True)

    print(f"--- Starting Training from Batch {start_batch_idx} ---")
    global_step = start_batch_idx
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"--- Starting Epoch {epoch+1}/{NUM_EPOCHS} ---")
        epoch_step = 0 

        for inputs, policy_targets, value_targets in loader:
            if epoch_step >= BATCHES_PER_EPOCH: break
            
            epoch_step += 1
            global_step += 1

            inputs = inputs.to(device, non_blocking=True)
            policy_targets = policy_targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            # --- KEY CHANGE: BFLOAT16 AUTOCAST ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                policy_logits, value_preds = model(inputs)
                loss_p = policy_loss_fn(policy_logits, policy_targets)
                loss_v = value_loss_fn(value_preds.squeeze(), value_targets.squeeze())
                total_loss = loss_p + loss_v

            # Safety check (still good practice)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"WARNING: NaN/Inf detected at step {global_step}. Skipping batch.")
                continue

            # NO SCALER, JUST BACKWARD
            total_loss.backward()
            
            # Clip gradients (still good for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            if global_step % 10 == 0: 
                current_lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step} | LR: {current_lr:.2e} | Loss: {total_loss.item():.4f} (P:{loss_p.item():.3f} V:{loss_v.item():.3f})")

            if global_step % 500 == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'batch_idx': global_step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, filename="latest_checkpoint.pth.tar")

        save_checkpoint({
            'epoch': epoch + 1,
            'batch_idx': global_step,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")

print("Training complete.")
if __name__ == "__main__":
    mp.set_sharing_strategy('file_system')
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
