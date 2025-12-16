"""
Climb Generation Model - Hands Only, Feature Space Output

Predicts the next hand hold position as a 5-dimensional feature vector:
[norm_x, norm_y, pull_x, pull_y, useability, type]

Input:  10 features (left hand + right hand, 5 features each)
Output: 5 features (predicted next hold in feature space)
"""

import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from tqdm import tqdm

# --- Constants ---
NUM_LIMBS = 2
FEATURE_DIM = 5
INPUT_DIM = NUM_LIMBS * FEATURE_DIM     # 2 hands × 5 features
OUTPUT_DIM = NUM_LIMBS * FEATURE_DIM    # Next position in Feature space.
HIDDEN_DIM = 128

NULL_FEATURES = [-1.0, -1.0, 0.0, 0.0, -1.0]

HOLDS_PATH = Path("data/holds_final.json")
CLIMBS_PATH = Path("data/spraywall-climbs.json")
DEVICE = "cpu"


# --- Data Utilities ---
def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def extract_hold_features(hold_data: dict) -> List[float]:
    """Extract normalized 5D feature vector from hold data."""
    if hold_data == -1:
        return list(NULL_FEATURES)
    
    return [
        float(hold_data['norm_x']),
        float(hold_data['norm_y']),
        float(hold_data['pull_x']),
        float(hold_data['pull_y']),
        float(hold_data['useability']) / 10.0
    ]


# --- Dataset ---
class ClimbDataset(Dataset):
    """
    Dataset of (current_hands, next_hold) pairs.
    
    For each transition t → t+1, identifies which hand moved and uses
    that hand's new position as the target.
    """
    
    def __init__(self, climbs_path: str, climb_indices: List[int]):
        self.examples = []
        data = load_json(climbs_path)
        
        for idx in climb_indices:
            self._process_climb(data['climbs'][idx])

    def _get_hand_features(self, position: dict) -> Tuple[List[float], List[float]]:
        """Extract left and right hand features from a position."""
        holds = position['holdsByLimb']
        return extract_hold_features(holds[0]), extract_hold_features(holds[1])

    def _process_climb(self, climb: dict):
        """Create (input, target) pairs from climb sequence."""
        sequence = climb['sequence']
        
        for t in range(len(sequence) - 1):
            lh_curr, rh_curr = self._get_hand_features(sequence[t])
            lh_next, rh_next = self._get_hand_features(sequence[t + 1])
            
            # Input: both hands' current features
            input_features = lh_curr + rh_curr
            
            # Target: the new position.
            target_features = lh_next + rh_next
            
            self.examples.append((
                torch.tensor(input_features, dtype=torch.float32),
                torch.tensor(target_features, dtype=torch.float32)
            ))

    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


# --- Model ---
class ClimbMLP(nn.Module):
    """
    Simple MLP: Input(10) → Dense(128) → ReLU → Dense(128) → ReLU → Output(5)
    """
    
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
def run_epoch(model: nn.Module, 
              loader: DataLoader, 
              criterion: nn.Module, 
              optimizer: Optional[optim.Optimizer], 
              device: str) -> Tuple[float, float]:
    """
    Run one training or validation epoch.
    Returns (avg_loss, avg_euclidean_distance).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    total_dist = 0.0
    n_samples = 0
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    
    with context:
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
            total_dist += torch.norm(outputs - targets, dim=1).sum().item()
            n_samples += len(inputs)

    return total_loss / n_samples, total_dist / n_samples


def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                num_epochs: int = 100, 
                lr: float = 0.001, 
                device: str = DEVICE) -> nn.Module:
    """Train with MSE loss, save best model by validation loss."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    pbar = tqdm(range(num_epochs), desc="Training")
    for _ in pbar:
        train_loss, train_dist = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dist = run_epoch(model, val_loader, criterion, None, device)
        
        pbar.set_postfix({
            "T_Dist": f"{train_dist:.4f}",
            "V_Dist": f"{val_dist:.3f}"
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_climb_mlp.pth')
            
    return model


# --- Inference ---
class ClimbGenerator:
    """Generate climb sequences by predicting holds in feature space."""
    
    def __init__(self, model: ClimbMLP, hold_map: Dict[int, List[float]], device: str = DEVICE):
        self.model = model.to(device).eval()
        self.hold_map = hold_map
        self.device = device

    def _to_input(self, lh: List[float], rh: List[float]) -> torch.Tensor:
        """Convert hand features to model input tensor."""
        return torch.tensor(lh + rh, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _nearest_hold(self, features: np.ndarray) -> Tuple[int, List[float]]:
        """Find hold with minimum Euclidean distance to predicted features."""
        best_id, best_dist = -1, float('inf')
        
        for hold_id, hold_features in self.hold_map.items():
            dist = np.linalg.norm(features - np.array(hold_features))
            if dist < best_dist:
                best_id, best_dist = hold_id, dist
                
        return best_id, list(self.hold_map.get(best_id))

    def generate(self, 
                 start_lh: List[float], 
                 start_rh: List[float], 
                 max_moves: int = 10) -> List[Tuple[int, int, List[float], List[float]]]:
        """
        Generate a climb sequence.
        
        Returns list of (lh_id, rh_id, lh_features, rh_features) tuples.
        """
        lh, rh = start_lh, start_rh
        sequence = []
        
        with torch.no_grad():
            for _ in range(max_moves):
                inputs = self._to_input(lh, rh)
                predicted = self.model(inputs).cpu().numpy().flatten()
                lh_id, lh_features = self._nearest_hold(predicted[:5])
                rh_id, rh_features = self._nearest_hold(predicted[5:])
                
                sequence.append((lh_id, rh_id))
                
                if lh_features == rh_features == NULL_FEATURES:
                    break
                    
        return sequence


# --- Data Loading Utilities ---
def load_hold_map(holds_path: str) -> Dict[int, List[float]]:
    """Load holds as {hold_id: feature_vector} mapping."""
    data = load_json(holds_path)
    return {h['hold_id']: extract_hold_features(h) for h in data['holds']}


def create_dataloaders(climbs_path: str,
                       train_indices: List[int], 
                       val_indices: List[int],
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_ds = ClimbDataset(climbs_path, train_indices)
    val_ds = ClimbDataset(climbs_path, val_indices)
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    )


def mirror_climb(sequence: np.ndarray) -> np.ndarray:
    """
    Data augmentation: Mirror a climb left-to-right.
    
    Args:
        sequence: np.ndarray of shape (num_moves,10)
    
    Returns:
        Mirrored sequence of same shape
    """
    mirrored = sequence.copy()
    
    # Swap LH <-> RH
    lh = mirrored[:, 0:5].copy()
    rh = mirrored[:, 5:].copy()
    mirrored[:, 0:5] = rh
    mirrored[:, 5:10] = lh
    
    # Negate pull_x, norm_x = 1 - norm_x
    for limb_offset in [0, 5]:
        # filter mask for the NULL_FEATURES. Do NOT 'mirror' these holds.
        # Non-null -> "(norm_x) is in the range [0,1]. Any hold with norm_x = -1 is a NULL_HOLD"
        non_null_holds = sequence[:,limb_offset] != -1
        mirrored[non_null_holds, limb_offset] = (1-mirrored[:, limb_offset])  # norm_x = 1 - norm_x
        mirrored[non_null_holds, limb_offset + 2] = -mirrored[:, limb_offset + 2]  # pull_x = -pull_x
    
    return mirrored

def translate_climb(sequence: np.ndarray) -> List[np.ndarray]:
    """
    Data Augmentation by translating a climb horizontally, while keeping x features within [0,1].
    
    :param sequence: The climb sequence to be augmented
    :type sequence: np.ndarray
    :param direction: The direction to shift the climb
    :type direction: np.ndarray
    :return: The augmented climb.
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """
    translated_left = sequence.copy()
    translated_right = sequence.copy()

    max_left = 1
    max_right = 1

    for limb_offset in [0,5]:
        non_null_holds = sequence[:,limb_offset]!=-1
        if np.any(non_null_holds):
            max_left = min(max_left,np.min(sequence[non_null_holds,limb_offset]))
            max_right = min(max_right,np.min(1-sequence[non_null_holds,limb_offset]))
    
    for limb_offset in [0,5]:
        non_null_holds = sequence[:,limb_offset]!=-1
        if np.any(non_null_holds):
            translated_left[non_null_holds,limb_offset] -= max_left
            translated_right[non_null_holds,limb_offset] += max_right
        assert np.max(translated_left[non_null_holds,limb_offset]) <= 1
        assert np.max(translated_right[non_null_holds,limb_offset]) <= 1
        assert np.min(translated_right[non_null_holds,limb_offset]) >= 0
        assert np.min(translated_right[non_null_holds,limb_offset]) >= 0

    return [translated_left, translated_right]

def augment_dataset(sequences: List[np.ndarray], names: List[str], grades: List[int]) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """
    Apply data augmentation to all climbs by mirroring, translating max-left, and translating max-right.
    
    Returns augmented dataset with 4x the original size.
    """
    aug_sequences = sequences.copy()
    aug_names = names.copy()
    aug_grades = grades.copy()
    
    for seq, name, grade in zip(sequences, names, grades):
        print(name)
        mirrored = mirror_climb(seq)
        left, right = translate_climb(seq)
        m_right, m_left = translate_climb(mirrored)
        # I just want to check that I've created a valid symmetric group and nothing fishy is going on.
        assert mirror_climb(left) == m_left
        aug_sequences.extend([seq, left, right, mirrored, m_right, m_left])
        aug_names.extend([f"{name}",f"{name} (left)",f"{name} (right)", f"{name} (mirrored)", f"{name} (right=>mirrored)", f"{name} (left=>mirrored)"])
        aug_grades.extend([[grade]*6])
    
    return aug_sequences, aug_names, aug_grades
