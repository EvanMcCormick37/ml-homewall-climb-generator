import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tqdm import tqdm

# --- Constants ---
NUM_LIMBS = 2
FEATURE_DIM = 5
INPUT_DIM = NUM_LIMBS * FEATURE_DIM     # 2 hands × 5 features
OUTPUT_DIM = NUM_LIMBS * FEATURE_DIM    # Next position in Feature space
HIDDEN_DIM = 128

NULL_FEATURES = [-1.0, -1.0, 0.0, 0.0, -1.0]

# --- Augmentation & Math Utilities ---

def mirror_climb(sequence: np.ndarray) -> np.ndarray:
    """
    Mirror a climb left-to-right. Swaps limbs and flips x-coordinates.
    Expects sequence shape (N, 10).
    """
    mirrored = sequence.copy()
    
    # 1. Swap LH (cols 0-4) and RH (cols 5-9)
    mirrored[:, [0, 1, 2, 3, 4]] = sequence[:, [5, 6, 7, 8, 9]]
    mirrored[:, [5, 6, 7, 8, 9]] = sequence[:, [0, 1, 2, 3, 4]]
    
    # 2. Invert norm_x (1-x) and pull_x (-x) for valid holds
    # norm_x is at offset 0, pull_x is at offset 2
    for limb_start_idx in [0, 5]:
        norm_x_idx = limb_start_idx
        pull_x_idx = limb_start_idx + 2
        
        # Mask: limb is not NULL (norm_x != -1)
        mask = mirrored[:, norm_x_idx] != -1
        
        mirrored[mask, norm_x_idx] = 1.0 - mirrored[mask, norm_x_idx]
        mirrored[mask, pull_x_idx] = -mirrored[mask, pull_x_idx]
        
    return mirrored


def translate_climb(sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate shifted max-left and shifted max-right variants.
    Ensures all x-coordinates stay within [0, 1].
    """
    # 1. Identify all valid X coordinates across both limbs
    lh_x = sequence[:, 0]
    rh_x = sequence[:, 5]
    
    valid_x_values = np.concatenate([lh_x[lh_x != -1], rh_x[rh_x != -1]])
    
    if valid_x_values.size == 0:
        return sequence.copy(), sequence.copy()

    # 2. Calculate max allowable shifts
    min_x = np.min(valid_x_values)
    max_x = np.max(valid_x_values)
    
    shift_left_val = min_x          # Amount to subtract to touch left wall
    shift_right_val = 1.0 - max_x   # Amount to add to touch right wall

    # 3. Apply shifts
    left_variant = sequence.copy()
    right_variant = sequence.copy()

    for limb_idx in [0, 5]:
        mask = sequence[:, limb_idx] != -1
        left_variant[mask, limb_idx] -= shift_left_val
        right_variant[mask, limb_idx] += shift_right_val

    return left_variant, right_variant


def augment_sequence_list(sequences: List[np.ndarray]) -> List[np.ndarray]:
    """
    Expands a list of sequences by 6x using mirroring and translation.
    """
    augmented = []
    for seq in sequences:
        # Generate mirrored variation
        mirrored = mirror_climb(seq)
        
        # Generate translations
        orig_l, orig_r = translate_climb(seq)
        mir_l, mir_r = translate_climb(mirrored)
        
        # Add all 6 variations
        augmented.extend([seq, orig_l, orig_r, mirrored, mir_l, mir_r])
        
    return augmented


def extract_hold_features(hold_data: dict) -> List[float]:
    """Extract normalized 5D feature vector from hold data dict."""
    if hold_data == -1:
        return list(NULL_FEATURES)
    
    return [
        float(hold_data['norm_x']),
        float(hold_data['norm_y']),
        float(hold_data['pull_x']),
        float(hold_data['pull_y']),
        float(hold_data['useability']) / 10.0
    ]


def parse_climb_to_numpy(climb_data: dict) -> np.ndarray:
    """
    Converts a raw climb dict into a (T, 10) numpy array of features.
    """
    sequence_features = []
    
    for position in climb_data['sequence']:
        holds = position['holdsByLimb']
        # Extract features for Left Hand (0) and Right Hand (1)
        lh_feat = extract_hold_features(holds[0])
        rh_feat = extract_hold_features(holds[1])
        sequence_features.append(lh_feat + rh_feat)
        
    return np.array(sequence_features, dtype=np.float32)


# --- Dataset ---
class ClimbDataset(Dataset):
    """
    Dataset of (current_hands, next_hold) pairs.
    Accepts a list of pre-processed (potentially augmented) numpy sequences.
    """
    def __init__(self, sequences: List[np.ndarray]):
        self.examples = []
        
        for seq in sequences:
            # seq shape is (T, 10)
            # Create pairs: Input(t) -> Target(t+1)
            for t in range(len(seq) - 1):
                input_feat = seq[t]
                target_feat = seq[t + 1]
                
                self.examples.append((
                    torch.tensor(input_feat, dtype=torch.float32),
                    torch.tensor(target_feat, dtype=torch.float32)
                ))

    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


# --- Data Pipeline ---
def load_and_preprocess_data(json_path: str, val_split: float = 0.2) -> Tuple[ClimbDataset, ClimbDataset]:
    """
    Loads JSON, converts to numpy, splits data, augments TRAINING set only,
    and returns ClimbDatasets.
    """
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_climbs = data['climbs']
    num_climbs = len(all_climbs)
    
    # 1. Convert all raw JSON climbs to Numpy arrays
    print("Parsing sequences...")
    all_sequences = [parse_climb_to_numpy(c) for c in all_climbs]
    
    # 2. Split indices
    indices = np.arange(num_climbs)
    np.random.shuffle(indices)
    split_idx = int(num_climbs * (1 - val_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_seqs = [all_sequences[i] for i in train_indices]
    val_seqs = [all_sequences[i] for i in val_indices]
    
    print(f"Split: {len(train_seqs)} Train / {len(val_seqs)} Val")
    
    # 3. Augment ONLY the training set
    print("Augmenting training set (6x)...")
    train_seqs_augmented = augment_sequence_list(train_seqs)
    print(f"Augmented Train size: {len(train_seqs_augmented)}")
    
    # 4. Create Datasets
    train_dataset = ClimbDataset(train_seqs_augmented)
    val_dataset = ClimbDataset(val_seqs)
    
    return train_dataset, val_dataset


# --- Model ---
class ClimbMLP(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --- Training ---
def run_epoch(
    model: nn.Module,
    loader: DataLoader, criterion: nn.Module, 
    optimizer: optim.Optimizer | None,
    device: str
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, total_dist, n_samples = 0.0, 0.0, 0
    
    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if is_train:
                optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if is_train:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * len(inputs)
            # Euclidean distance for monitoring
            total_dist += torch.norm(outputs - targets, dim=1).sum().item()
            n_samples += len(inputs)

    return total_loss / n_samples, total_dist / n_samples


def train_model(train_ds: Dataset, val_ds: Dataset, 
                num_epochs: int = 100, lr: float = 0.001, 
                batch_size: int = 32, device: str = "cpu") -> nn.Module:
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = ClimbMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    pbar = tqdm(range(num_epochs), desc="Training")
    for _ in pbar:
        train_loss, train_dist = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dist = run_epoch(model, val_loader, criterion, None, device)
        
        pbar.set_postfix({"T_MSE": f"{train_loss:.4f}", "V_MSE": f"{val_loss:.4f}"})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_climb_mlp.pth')
            
    return model