import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tqdm import tqdm

# --- Constants ---
PATH = 'data/all-data.json'
NUM_LIMBS = 2
FEATURE_DIM = 5
INPUT_DIM = NUM_LIMBS * FEATURE_DIM     # 2 hands × 5 features
OUTPUT_DIM = NUM_LIMBS * FEATURE_DIM    # Next position in Feature space
HIDDEN_DIM = 256

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

def parse_climb_to_numpy(climb_data: dict, hold_map: dict[int, List[float]]) -> np.ndarray:
    """
    Converts a raw climb dict into a (T, 10) numpy array of features.
    """
    sequence_features = []
    
    for position in climb_data['sequence']:
        feature_list=[]
        for hold_idx in position:
            if hold_idx == -1:
                feature_list.append(NULL_FEATURES)
            else:
                feature_list.append(hold_map[hold_idx])
        
        # Combine features (Left + Right)
        sequence_features.append(feature_list[0] + feature_list[1])
    # Append NULL, NULL as a <STOP> token to alert the nn that the sequence is done.
    sequence_features.append(NULL_FEATURES+NULL_FEATURES)
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
def load_and_preprocess_data(path: str, val_split: float = 0.2, sequential: bool = False) -> Tuple[ClimbDataset, ClimbDataset, dict[int, List[float]]]:
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
    print(f"{int(data['metadata']['num_moves']) * 6} training moves estimated with dataset augmentation...")

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


# --- Models ---
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
                        nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ClimbRNN(nn.Module):
    """
    Vanilla RNN for climb sequence prediction.
    
    Compatible with existing ClimbGenerator and training loop via forward().
    Use forward_sequence() for full sequence processing with hidden states.
    """
    
    def __init__(self, 
                 input_dim: int = INPUT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 output_dim: int = OUTPUT_DIM,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity='tanh'
        )
        
        # Output projection layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single-step forward pass (MLP-compatible interface).
        
        Args:
            x: Input tensor of shape (batch, input_dim)
        
        Returns:
            output: Predictions of shape (batch, output_dim)
        """
        # Add sequence dimension: (batch, input_dim) -> (batch, 1, input_dim)
        x = x.unsqueeze(1)
        batch_size = x.size(0)
        
        # Initialize fresh hidden state for each forward pass
        hidden = self.init_hidden(batch_size, x.device)
        
        # RNN forward
        rnn_out, _ = self.rnn(x, hidden)  # (batch, 1, hidden_dim)
        
        # Project to output space
        output = self.fc(rnn_out.squeeze(1))  # (batch, output_dim)
        
        return output
    
    def forward_sequence(self, 
                         x: torch.Tensor, 
                         hidden: torch.Tensor | None = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full sequence forward pass with hidden state management.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional hidden state of shape (num_layers, batch, hidden_dim)
        
        Returns:
            output: Predictions of shape (batch, seq_len, output_dim)
            hidden: Final hidden state of shape (num_layers, batch, hidden_dim)
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # RNN processes entire sequence
        rnn_out, hidden = self.rnn(x, hidden)  # (batch, seq_len, hidden_dim)
        
        # Project each timestep to output space
        output = self.fc(rnn_out)  # (batch, seq_len, output_dim)
        
        return output, hidden
    
    def forward_step(self,
                     x: torch.Tensor,
                     hidden: torch.Tensor | None = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step with explicit hidden state (for autoregressive generation).
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            hidden: Hidden state of shape (num_layers, batch, hidden_dim)
        
        Returns:
            output: Predictions of shape (batch, output_dim)
            hidden: Updated hidden state
        """
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        rnn_out, hidden = self.rnn(x, hidden)
        output = self.fc(rnn_out.squeeze(1))
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state with zeros."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, 
            device=device
        )

class ClimbLSTM(nn.Module):
    """
    LSTM for climb sequence prediction.
    
    Includes cell state for better long-term dependency modeling.
    Compatible with existing ClimbGenerator and training loop via forward().
    Use forward_sequence() for full sequence processing with hidden states.
    """
    
    def __init__(self,
                 input_dim: int = INPUT_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 output_dim: int = OUTPUT_DIM,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single-step forward pass (MLP-compatible interface).
        
        Args:
            x: Input tensor of shape (batch, input_dim)
        
        Returns:
            output: Predictions of shape (batch, output_dim)
        """
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        batch_size = x.size(0)
        
        # Initialize fresh hidden and cell states
        hidden = self.init_hidden(batch_size, x.device)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x, hidden)  # (batch, 1, hidden_dim)
        
        # Project to output space
        output = self.fc(lstm_out.squeeze(1))  # (batch, output_dim)
        
        return output
    
    def forward_sequence(self,
                         x: torch.Tensor,
                         hidden: Tuple[torch.Tensor, torch.Tensor] | None = None
                         ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full sequence forward pass with hidden state management.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional tuple of (h_0, c_0), each of shape 
                    (num_layers, batch, hidden_dim)
        
        Returns:
            output: Predictions of shape (batch, seq_len, output_dim)
            hidden: Tuple of final (h_n, c_n) states
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # LSTM processes entire sequence
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq_len, hidden_dim)
        
        # Project each timestep to output space
        output = self.fc(lstm_out)  # (batch, seq_len, output_dim)
        
        return output, hidden
    
    def forward_step(self,
                     x: torch.Tensor,
                     hidden: Tuple[torch.Tensor, torch.Tensor] | None = None
                     ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single step with explicit hidden state (for autoregressive generation).
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            hidden: Tuple of (h, c), each of shape (num_layers, batch, hidden_dim)
        
        Returns:
            output: Predictions of shape (batch, output_dim)
            hidden: Updated (h, c) tuple
        """
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out.squeeze(1))
        
        return output, hidden
    
    def init_hidden(self, 
                    batch_size: int, 
                    device: torch.device
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state (h_0) and cell state (c_0) with zeros."""
        h_0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim,
            device=device
        )
        return (h_0, c_0)


# --- Autoregressive Generation ---
class ClimbGenerator:
    """Generate climb sequences by predicting holds in feature space."""
    
    def __init__(self, model: ClimbMLP, hold_map: dict[int, List[float]], device: str):
        self.model = model.to(device).eval()
        self.hold_map = hold_map
        self.device = device

    def _to_input(self, lh: List[float], rh: List[float]) -> torch.Tensor:
        """Convert hand features to model input tensor."""
        return torch.tensor(lh + rh, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _nearest_hold(self, features: np.ndarray) -> Tuple[int, List[float], int]:
        """Find hold with minimum Euclidean distance to predicted features."""
        best_id, best_dist = -1, float('inf')
        
        for hold_id, hold_features in self.hold_map.items():
            dist = np.linalg.norm(features - np.array(hold_features))
            if dist < best_dist:
                best_id, best_dist = hold_id, dist
                
        return best_id, list(self.hold_map.get(best_id, NULL_FEATURES)), best_dist

    def generate(self, 
                 start_lh: int, 
                 start_rh: int, 
                 max_moves: int = 10) -> List[Tuple[int, int, List[float], List[float]]]:
        """
        Generate a climb sequence from starting hold indices.
        
        Returns list of (lh_id, rh_id, lh_features, rh_features) tuples.
        """
        lh_id = start_lh
        rh_id = start_rh
        lh = self.hold_map[start_lh]
        rh = self.hold_map[start_rh]
        sequence = []
        
        with torch.no_grad():
            for move in range(max_moves):
                inputs = self._to_input(lh, rh)
                predicted = self.model(inputs).cpu().numpy().flatten()
                lh_id_next, lh_next, lh_dist = self._nearest_hold(predicted[:5])
                rh_id_next, rh_next, rh_dist = self._nearest_hold(predicted[5:])

                if lh_id_next != lh_id and rh_id_next != rh_id:
                    if lh_dist < rh_dist:
                        lh_id = lh_id_next
                        lh = lh_next
                    else:
                        rh_id = rh_id_next
                        rh = rh_next
                else:
                    lh, rh = lh_next, rh_next
                    lh_id, rh_id = lh_id_next, rh_id_next
                
                if (len(sequence) > 0 and (lh_id, rh_id) == sequence[-1]) or (lh == NULL_FEATURES and rh == NULL_FEATURES):
                    break

                sequence.append((lh_id, rh_id))
                    
        return sequence

class ClimbGeneratorRNN:
    """Generate climb sequences using RNN with persistent hidden state."""
    
    def __init__(self, model: ClimbRNN, hold_map: dict[int, List[float]], device: str):
        self.model = model.to(device).eval()
        self.hold_map = hold_map
        self.device = device

    def _to_input(self, lh: List[float], rh: List[float]) -> torch.Tensor:
        """Convert hand features to model input tensor."""
        return torch.tensor(lh + rh, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _nearest_hold(self, features: np.ndarray) -> Tuple[int, List[float], float]:
        """Find hold with minimum Euclidean distance to predicted features."""
        best_id, best_dist = -1, float('inf')
        
        for hold_id, hold_features in self.hold_map.items():
            dist = np.linalg.norm(features - np.array(hold_features))
            if dist < best_dist:
                best_id, best_dist = hold_id, dist
                
        return best_id, list(self.hold_map.get(best_id, NULL_FEATURES)), best_dist

    def generate(self, 
                 start_lh: int, 
                 start_rh: int, 
                 max_moves: int = 10) -> List[Tuple[int, int]]:
        """
        Generate a climb sequence from starting hold indices.
        
        Returns list of (lh_id, rh_id) tuples.
        """
        lh_id = start_lh
        rh_id = start_rh
        lh = self.hold_map[start_lh]
        rh = self.hold_map[start_rh]
        sequence = []
        
        # Initialize hidden state once at the start
        hidden = self.model.init_hidden(batch_size=1, device=self.device)
        
        with torch.no_grad():
            for move in range(max_moves):
                inputs = self._to_input(lh, rh)
                
                # Use forward_step to maintain hidden state across moves
                predicted, hidden = self.model.forward_step(inputs, hidden)
                predicted = predicted.cpu().numpy().flatten()
                
                lh_id_next, lh_next, lh_dist = self._nearest_hold(predicted[:5])
                rh_id_next, rh_next, rh_dist = self._nearest_hold(predicted[5:])

                if lh_id_next != lh_id and rh_id_next != rh_id:
                    if lh_dist < rh_dist:
                        lh_id = lh_id_next
                        lh = lh_next
                    else:
                        rh_id = rh_id_next
                        rh = rh_next
                else:
                    lh, rh = lh_next, rh_next
                    lh_id, rh_id = lh_id_next, rh_id_next
                
                if (len(sequence) > 0 and (lh_id, rh_id) == sequence[-1]) or (lh == NULL_FEATURES and rh == NULL_FEATURES):
                    break

                sequence.append((lh_id, rh_id))
                    
        return sequence


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

def collate_sequences(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences to the same length within a batch."""
    inputs, targets = zip(*batch)
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0.0)
    return inputs_padded, targets_padded

def run_epoch_sequential(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: str
) -> Tuple[float, float]:
    """Training loop for sequence models (RNN/LSTM)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, total_dist, n_samples = 0.0, 0.0, 0
    
    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if is_train:
                optimizer.zero_grad()
            
            outputs, _ = model.forward_sequence(inputs)
            
            # Flatten for loss: (batch * seq_len, 10)
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1, targets.size(-1))
            
            loss = criterion(outputs_flat, targets_flat)
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            n_timesteps = targets_flat.size(0)
            total_loss += loss.item() * n_timesteps
            total_dist += torch.norm(outputs_flat - targets_flat, dim=1).sum().item()
            n_samples += n_timesteps

    return total_loss / n_samples, total_dist / n_samples

def train_climb_generator_rnn(
    train_ds: ClimbSequenceDataset,
    val_ds: ClimbSequenceDataset,
    hold_map: dict[int, List[float]],
    num_epochs: int = 200,
    lr: float = 0.001,
    batch_size: int = 32,
    device: str = "cpu"
) -> ClimbGeneratorRNN:
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    
    model = ClimbRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    pbar = tqdm(range(num_epochs), desc="Training RNN")
    for _ in pbar:
        train_loss, train_dist = run_epoch_sequential(model, train_loader, criterion, optimizer, device)
        val_loss, val_dist = run_epoch_sequential(model, val_loader, criterion, None, device)
        
        pbar.set_postfix({"T_MSE": f"{train_loss:.4f}", "V_MSE": f"{val_loss:.4f}"})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_climb_rnn.pth')
            
    return ClimbGeneratorRNN(model, hold_map, device)

def train_climb_generator(
    train_ds: Dataset,
    val_ds: Dataset,
    hold_map: dict[int,List[float]],
    num_epochs: int = 100,
    lr: float = 0.001, 
    batch_size: int = 32,
    device: str = "cpu"
    ) -> ClimbGenerator:
    
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
            
    return ClimbGenerator(model, hold_map, device)

