import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple
from app.config import settings

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
        return list(settings.NULL_FEATURES)
    
    return [
        float(hold_data['norm_x']),
        float(hold_data['norm_y']),
        float(hold_data['pull_x']),
        float(hold_data['pull_y']),
        float(hold_data['useability']) / 10.0
    ]

def parse_climb_to_numpy(climb_data: dict, hold_map: dict[int, List[float]]) -> np.ndarray:
    """
    Converts a raw climb dict into a (T, 10) numpy array of features.
    """
    sequence_features = []
    
    for position in climb_data['sequence']:
        feature_list=[]
        for hold_idx in position:
            if hold_idx == -1:
                feature_list.append(settings.NULL_FEATURES)
            else:
                feature_list.append(hold_map[hold_idx])
        
        # Combine features (Left + Right)
        sequence_features.append(feature_list[0] + feature_list[1])
    # Append NULL, NULL as a <STOP> token to alert the nn that the sequence is done.
    # sequence_features.append(settings.NULL_FEATURES+settings.NULL_FEATURES)
    return np.array(sequence_features, dtype=np.float32)

def parse_moveset_to_numpy(moveset: dict, hold_map: dict[int, List[float]]) -> List[np.ndarray]:
    """
    Converts a moveset JSON into a list of sequence pairs.
    """
    moves = []
    lh_start = hold_map[moveset["lh_start"]]
    rh_start = hold_map[moveset["rh_start"]]

    for h in moveset['lh_finish']:
        lh_end = hold_map[h]
        moves.append(np.array([lh_start+rh_start,lh_end+rh_start],dtype=np.float32))
    for h in moveset['rh_finish']:
        rh_end = hold_map[h]
        moves.append(np.array([lh_start+rh_start,lh_start+rh_end],dtype=np.float32))

    return moves


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

class ClimbSequenceDataset(Dataset):
    """
    Dataset that returns full sequences for RNN training.
    Each item is (input_sequence, target_sequence) where target is shifted by 1.
    """
    def __init__(self, sequences: List[np.ndarray]):
        # Filter out sequences that are too short
        self.sequences = [seq for seq in sequences if len(seq) >= 2]
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        # Input: all positions except last
        # Target: all positions except first (shifted by 1)
        inputs = torch.tensor(seq[:-1], dtype=torch.float32)
        targets = torch.tensor(seq[1:], dtype=torch.float32)
        return inputs, targets


# --- Data Pipeline ---
def process_training_data(path: str, val_split: float = 0.2, sequential: bool = False) -> Tuple[ClimbDataset, ClimbDataset, dict[int, List[float]]]:
    """
    Loads JSON, converts to numpy, splits data, augments TRAINING set only,
    and returns ClimbDatasets.
    """
    print(f"Loading data from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    
    hold_map = {h['hold_id']: extract_hold_features(h) for h in data['holds']}
    all_sequences = []
    print(f"Extracted {len(hold_map)} holds...")

    all_sequences.extend(augment_sequence_list([parse_climb_to_numpy(s,hold_map) for s in data['sequences']]))
    if not sequential:
        all_sequences.extend([seq for moveset in data['movesets'] for seq in parse_moveset_to_numpy(moveset,hold_map)])
    num_sequences = len(all_sequences)
    print(f"Extracted {num_sequences} sequences...")
    print(f"{num_sequences * 4} training moves estimated with dataset augmentation...")

    # 2. Split indices
    indices = np.arange(num_sequences)
    np.random.shuffle(indices)
    split_idx = int(num_sequences * (1 - val_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_seqs = [all_sequences[i] for i in train_indices]
    val_seqs = [all_sequences[i] for i in val_indices]
    
    print(f"Split: {len(train_seqs)} Train / {len(val_seqs)} Val")
    
    # 4. Create Datasets
    if sequential:
        train_dataset = ClimbSequenceDataset(train_seqs)
        val_dataset = ClimbSequenceDataset(val_seqs)
    else:     
        train_dataset = ClimbDataset(train_seqs)
        val_dataset = ClimbDataset(val_seqs)
    
    return train_dataset, val_dataset, hold_map
