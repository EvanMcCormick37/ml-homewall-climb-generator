"""
Utilities for preparing training data from wall/climb data.

Handles:
- Feature extraction from holds
- Sequence parsing and augmentation
- Dataset creation for MLP and sequential models
"""
import json
import torch
import numpy as np
from torch.utils.data import Dataset

from app.database import WALLS_DIR, get_db
from app.schemas import FeatureConfig


# Feature dimensions for each feature type
FEATURE_DIMS = {
    "position": 2,       # norm_x, norm_y
    "pull_direction": 2, # pull_x, pull_y
    "difficulty": 1,     # useability (normalized)
}


def get_feature_dim(config: FeatureConfig) -> int:
    """Calculate total feature dimension based on config."""
    dim = 0
    if config.position:
        dim += FEATURE_DIMS["position"]
    if config.pull_direction:
        dim += FEATURE_DIMS["pull_direction"]
    if config.difficulty:
        dim += FEATURE_DIMS["difficulty"]
    return dim


def get_null_features(config: FeatureConfig) -> list[float]:
    """Get null feature vector (for missing/off-wall limbs)."""
    features = []
    if config.position:
        features.extend([-1.0, -1.0])
    if config.pull_direction:
        features.extend([0.0, 0.0])
    if config.difficulty:
        features.append(0.0)
    return features


def extract_hold_features(hold_data: dict, config: FeatureConfig) -> list[float]:
    """
    Extract feature vector from hold data based on config.
    
    Args:
        hold_data: Hold dict with norm_x, norm_y, pull_x, pull_y, useability
        config: Which features to include
        
    Returns:
        Feature vector of length get_feature_dim(config)
    """
    features = []
    
    if config.position:
        features.extend([
            float(hold_data["norm_x"]),
            float(hold_data["norm_y"]),
        ])
    
    if config.pull_direction:
        features.extend([
            float(hold_data["pull_x"]),
            float(hold_data["pull_y"]),
        ])
    
    if config.difficulty:
        # Normalize useability from 0-10 to 0-1
        features.append(float(hold_data["useability"]) / 10.0)
    
    return features


def build_hold_map(
    wall_id: str, 
    config: FeatureConfig,
) -> dict[int, list[float]]:
    """
    Load holds from wall JSON and build hold_id -> feature vector map.
    
    Args:
        wall_id: The wall ID
        config: Feature configuration
        
    Returns:
        Dict mapping hold_id to feature vector
    """
    wall_json_path = WALLS_DIR / wall_id / "wall.json"
    
    with open(wall_json_path, "r") as f:
        wall_data = json.load(f)
    
    hold_map = {}
    for hold in wall_data["holds"]:
        hold_id = hold["hold_id"]
        hold_map[hold_id] = extract_hold_features(hold, config)
    
    return hold_map


def load_climbs_from_db(wall_id: str) -> list[list[tuple]]:
    """
    Load all climbs for a wall from the database.
    
    Args:
        wall_id: The wall ID
        
    Returns:
        List of climb dicts with 'id' and 'sequence' keys
    """
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, sequence FROM climbs WHERE wall_id = ?",
            (wall_id,),
        ).fetchall()
    
    return [json.loads(row["sequence"]) for row in rows]


# --- Sequence Conversion ---

def parse_climb_to_numpy(
    climb: list[tuple[int]],
    hold_map: dict[int, list[float]],
    null_features: list[float],
) -> np.ndarray:
    """
    Convert a climb dict to a (T, num_features * num_limbs) numpy array.
    
    Args:
        climb: Dict with 'sequence' key containing [[lh, rh], ...]
        hold_map: hold_id -> feature vector mapping
        null_features: Feature vector for off-wall limbs
        
    Returns:
        Array of shape (num_positions, feature_dim * 2)
    """
    sequence_features = []
    
    for position in climb:
        lh_id, rh_id = position
        
        # Get features for each limb (-1 means off-wall)
        lh_features = null_features if lh_id == -1 else hold_map[lh_id]
        rh_features = null_features if rh_id == -1 else hold_map[rh_id]
        
        # Concatenate: [lh_features..., rh_features...]
        combined = lh_features + rh_features
        sequence_features.append(combined)
    
    return np.array(sequence_features, dtype=np.float32)


# --- Augmentation ---

def mirror_climb(
    sequence: np.ndarray, 
    feature_dim: int,
    config: FeatureConfig,
) -> np.ndarray:
    """
    Mirror a climb left-to-right by swapping limbs and flipping x-coordinates.
    
    Args:
        sequence: Array of shape (T, feature_dim * 2)
        feature_dim: Features per limb
        config: Feature config to know which indices to flip
        
    Returns:
        Mirrored sequence
    """
    mirrored = sequence.copy()
    
    # Swap LH and RH features
    lh_slice = slice(0, feature_dim)
    rh_slice = slice(feature_dim, feature_dim * 2)
    
    mirrored[:, lh_slice] = sequence[:, rh_slice]
    mirrored[:, rh_slice] = sequence[:, lh_slice]
    
    # Flip x-coordinates if position features are included
    if config.position:
        # norm_x is at index 0 for each limb
        for limb_start in [0, feature_dim]:
            norm_x_idx = limb_start
            mask = mirrored[:, norm_x_idx] != -1  # Valid holds only
            mirrored[mask, norm_x_idx] = 1.0 - mirrored[mask, norm_x_idx]
            
            # pull_x is at index 2 if pull_direction is enabled
            if config.pull_direction:
                pull_x_idx = limb_start + 2
                mirrored[mask, pull_x_idx] = -mirrored[mask, pull_x_idx]
    
    return mirrored


def translate_climb(
    sequence: np.ndarray,
    feature_dim: int,
    config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate max-left and max-right shifted variants.
    
    Args:
        sequence: Array of shape (T, feature_dim * 2)
        feature_dim: Features per limb
        config: Feature config
        
    Returns:
        Tuple of (left_shifted, right_shifted) sequences
    """
    if not config.position:
        # Can't translate without position features
        return sequence.copy(), sequence.copy()
    
    # Collect all valid x-coordinates
    lh_x = sequence[:, 0]
    rh_x = sequence[:, feature_dim]
    
    valid_x = np.concatenate([
        lh_x[lh_x != -1],
        rh_x[rh_x != -1],
    ])
    
    if valid_x.size == 0:
        return sequence.copy(), sequence.copy()
    
    # Calculate shift amounts
    min_x = np.min(valid_x)
    max_x = np.max(valid_x)
    shift_left = min_x
    shift_right = 1.0 - max_x
    
    # Apply shifts
    left_variant = sequence.copy()
    right_variant = sequence.copy()
    
    for limb_start in [0, feature_dim]:
        mask = sequence[:, limb_start] != -1
        left_variant[mask, limb_start] -= shift_left
        right_variant[mask, limb_start] += shift_right
    
    return left_variant, right_variant


def augment_sequences(
    sequences: list[np.ndarray],
    feature_dim: int,
    config: FeatureConfig,
) -> list[np.ndarray]:
    """
    Expand sequences 6x using mirroring and translation.
    
    Args:
        sequences: List of sequence arrays
        feature_dim: Features per limb
        config: Feature config
        
    Returns:
        Augmented list (6x original size)
    """
    augmented = []
    
    for seq in sequences:
        # Original + translations
        orig_left, orig_right = translate_climb(seq, feature_dim, config)
        
        # Mirrored + translations
        mirrored = mirror_climb(seq, feature_dim, config)
        mir_left, mir_right = translate_climb(mirrored, feature_dim, config)
        
        augmented.extend([
            seq,
            orig_left,
            orig_right,
            mirrored,
            mir_left,
            mir_right,
        ])
    
    return augmented


# --- Datasets ---

class ClimbDataset(Dataset):
    """
    Dataset of (current_position, next_position) pairs for MLP training.
    Flattens sequences into individual transition examples.
    """
    
    def __init__(self, sequences: list[np.ndarray]):
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        
        for seq in sequences:
            # Create pairs: position[t] -> position[t+1]
            for t in range(len(seq) - 1):
                input_feat = torch.tensor(seq[t], dtype=torch.float32)
                target_feat = torch.tensor(seq[t + 1], dtype=torch.float32)
                self.examples.append((input_feat, target_feat))
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


class ClimbSequenceDataset(Dataset):
    """
    Dataset of full sequences for RNN/LSTM training.
    Each item is (input_sequence[:-1], target_sequence[1:]).
    """
    
    def __init__(self, sequences: list[np.ndarray], min_length: int = 2):
        # Filter sequences that are too short
        self.sequences = [seq for seq in sequences if len(seq) >= min_length]
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        # Input: all except last, Target: all except first
        inputs = torch.tensor(seq[:-1], dtype=torch.float32)
        targets = torch.tensor(seq[1:], dtype=torch.float32)
        return inputs, targets


# --- Main Pipeline ---

def process_training_data(
    wall_id: str,
    config: FeatureConfig,
    sequential: bool = False,
    augment: bool = True,
    val_split: float = 0.2,
) -> tuple[Dataset, Dataset, dict[int, list[float]], int]:
    """
    Load and preprocess training data for a wall.
    
    Args:
        wall_id: The wall ID
        config: Feature configuration
        sequential: If True, return sequence datasets for RNN/LSTM
        augment: If True, augment training data (6x)
        val_split: Fraction of data for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset, hold_map, num_climbs)
    """
    # Load hold features
    hold_map = build_hold_map(wall_id, config)
    null_features = get_null_features(config)
    feature_dim = get_feature_dim(config)
    
    # Load climbs from database
    climbs = load_climbs_from_db(wall_id)
    num_climbs = len(climbs)
    num_moves = sum([len(c) for c in climbs])
    
    if num_climbs == 0:
        raise ValueError(f"No climbs found for wall {wall_id}")
    
    # Convert to numpy sequences
    sequences = [
        parse_climb_to_numpy(climb, hold_map, null_features)
        for climb in climbs
    ]
    
    # Filter out very short sequences
    sequences = [seq for seq in sequences if len(seq) >= 2]
    
    # Split into train/val BEFORE augmentation
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    split_idx = int(len(sequences) * (1 - val_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_seqs = [sequences[i] for i in train_indices]
    val_seqs = [sequences[i] for i in val_indices]
    
    # Augment training data only
    if augment:
        train_seqs = augment_sequences(train_seqs, feature_dim, config)
    
    # Create appropriate dataset type
    if sequential:
        train_dataset = ClimbSequenceDataset(train_seqs)
        val_dataset = ClimbSequenceDataset(val_seqs)
    else:
        train_dataset = ClimbDataset(train_seqs)
        val_dataset = ClimbDataset(val_seqs)
    
    return train_dataset, val_dataset, hold_map, num_climbs, num_moves