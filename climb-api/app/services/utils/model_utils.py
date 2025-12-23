import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from app.config import settings
from app.schemas import ModelType, FeatureConfig
from app.services.utils.train_data_utils import get_feature_dim, get_null_features

# --- Models ---
class ClimbMLP(nn.Module):
    """Simple multilayer perceptron"""
    def __init__(self,
                 feature_config: FeatureConfig,
                 num_limbs: int = settings.NUM_LIMBS,
                 hidden_dim: int = settings.HIDDEN_DIM,
                 n_hidden_layers: int = settings.N_HIDDEN_LAYERS,
                 ):
        super().__init__()
        self.is_sequential = False
        self.num_limbs = num_limbs
        self.feature_config = feature_config
        self.num_features = get_feature_dim(feature_config)
        self.null_features = get_null_features(feature_config)

        feature_dim = self.num_limbs * self.num_features
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            )
        for _ in range(n_hidden_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())
        
        self.net.append(
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
# class ClimbMLPSoftmax(nn.Module):
#     def __init__(self,
#                  output_dim = settings.NUM_LIMBS * settings.NUM_FEATURES,
#                  input_dim: int = settings.NUM_LIMBS * settings.NUM_FEATURES,
#                  hidden_dim: int = settings.HIDDEN_DIM,
#                  ):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#             nn.ReLU(),
#             nn.Softmax(output_dim)
#         )
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)

class ClimbSequential(nn.Module):
    """
    Base class for sequential climb models (RNN, LSTM).
    
    Subclasses must:
    1. Call super().__init__(...) 
    2. Set self.seq_layer
    3. Set self.fc (output projection layer)
    4. Implement init_hidden()
    """
    
    def __init__(self,
                 feature_config: FeatureConfig,
                 num_limbs: int = settings.NUM_LIMBS,
                 hidden_dim: int = settings.HIDDEN_DIM,
                 n_hidden_layers: int = settings.N_HIDDEN_LAYERS,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        self.is_sequential = True
        self.num_limbs = num_limbs
        self.hidden_dim = hidden_dim
        self.num_layers = n_hidden_layers
        self.dropout = dropout
        self.feature_config = feature_config
        self.num_features = get_feature_dim(feature_config)
        self.null_features = get_null_features(feature_config)
        self.feature_dim = self.num_limbs * self.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single-step forward pass (MLP-compatible interface).
        
        Args:
            x: Input tensor of shape (batch, feature_dim)
        
        Returns:
            output: Predictions of shape (batch, feature_dim)
        """
        x = x.unsqueeze(1)  # (batch, 1, feature_dim)
        batch_size = x.size(0)
        
        hidden = self.init_hidden(batch_size, x.device)
        seq_out, _ = self.seq_layer(x, hidden)  # (batch, 1, hidden_dim)
        output = self.fc(seq_out.squeeze(1))  # (batch, feature_dim)
        
        return output
    
    def forward_sequence(self, 
                         x: torch.Tensor, 
                         hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """
        Full sequence forward pass with hidden state management.
        
        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim)
            hidden: Optional initial hidden state
        
        Returns:
            output: Predictions of shape (batch, seq_len, feature_dim)
            hidden: Final hidden state
        """
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        seq_out, hidden = self.seq_layer(x, hidden)  # (batch, seq_len, hidden_dim)
        output = self.fc(seq_out)  # (batch, seq_len, feature_dim)
        
        return output, hidden
    
    def forward_step(self,
                     x: torch.Tensor,
                     hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """
        Single step with explicit hidden state (for autoregressive generation).
        
        Args:
            x: Input tensor of shape (batch, feature_dim)
            hidden: Hidden state from previous step
        
        Returns:
            output: Predictions of shape (batch, feature_dim)
            hidden: Updated hidden state
        """
        x = x.unsqueeze(1)  # (batch, 1, feature_dim)
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        seq_out, hidden = self.seq_layer(x, hidden)
        output = self.fc(seq_out.squeeze(1))
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement init_hidden")


class ClimbRNN(ClimbSequential):
    """
    Vanilla RNN for climb sequence prediction.
    
    Uses a single hidden state tensor.
    """
    
    def __init__(self,
                 feature_config: FeatureConfig,
                 num_limbs: int = settings.NUM_LIMBS,
                 hidden_dim: int = settings.HIDDEN_DIM,
                 n_hidden_layers: int = settings.N_HIDDEN_LAYERS,
                 dropout: float = 0.1,
                 ):
        super().__init__(feature_config, num_limbs, hidden_dim, n_hidden_layers, dropout)
        
        self.seq_layer = nn.RNN(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(self.hidden_dim, self.feature_dim)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state with zeros."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, 
            device=device
        )


class ClimbLSTM(ClimbSequential):
    """
    LSTM for climb sequence prediction.
    
    Uses a tuple of (hidden_state, cell_state) tensors.
    """
    
    def __init__(self,
                 feature_config: FeatureConfig,
                 num_limbs: int = settings.NUM_LIMBS,
                 hidden_dim: int = settings.HIDDEN_DIM,
                 n_hidden_layers: int = settings.N_HIDDEN_LAYERS,
                 dropout: float = 0.1,
                 ):
        super().__init__(feature_config, num_limbs, hidden_dim, n_hidden_layers, dropout)
        
        self.seq_layer = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(self.hidden_dim, self.feature_dim)
    
    def init_hidden(self, 
                    batch_size: int, 
                    device: torch.device
                    ) -> tuple[torch.Tensor, torch.Tensor]:
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
 

def create_model_instance(
    model_type: ModelType,
    feature_config: FeatureConfig,
    ) -> nn.Module:
    """Create a fresh (untrained) model according to specifications"""
    match model_type:
        case ModelType.MLP:
            return ClimbMLP(feature_config)
        case ModelType.RNN:
            return ClimbRNN(feature_config)
        case ModelType.LSTM:
            return ClimbLSTM(feature_config)
        case _:
            raise TypeError(f"Invalid model type: {model_type}")


# --- Training ---
def collate_sequences(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences to the same length within a batch."""
    inputs, targets = zip(*batch)
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0.0)
    return inputs_padded, targets_padded

def run_epoch_mlp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
    device: str = settings.DEVICE
) -> tuple[float, float]:
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

def run_epoch_sequential(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
    device: str = settings.DEVICE,
) -> tuple[float, float]:
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

def run_epoch(
    model: nn.Module,
    is_sequential: bool,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
    device: str = settings.DEVICE,
) -> tuple[float,float]:
    if is_sequential:
        return run_epoch_sequential(model,loader,criterion,optimizer,device)
    else:
        return run_epoch_mlp(model,loader,criterion,optimizer,device)


# --- Autoregressive Generation ---
class ClimbGenerator:
    """Generate climb sequences by predicting holds in feature space."""
    
    def __init__(self, model: ClimbMLP, hold_map: dict[int, list[float]], device: str):
        self.model = model.to(device).eval()
        self.hold_map = hold_map
        self.device = device

    def _to_input(self, lh: list[float], rh: list[float]) -> torch.Tensor:
        """Convert hand features to model input tensor."""
        return torch.tensor(lh + rh, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _sample_hold(self, features: np.ndarray, temperature: float) -> tuple[int, list[float]]:
        """Sample a hold with probability inversely proportional to distance."""
        hold_ids = list(self.hold_map.keys())
        distances = np.array([
            np.linalg.norm(features - np.array(self.hold_map[hid])[:len(features)]) 
            for hid in hold_ids
        ])
        
        # Convert distances to probabilities (lower distance = higher prob)
        # Using negative distance as logits
        logits = -distances / temperature
        probs = np.exp(logits - np.max(logits))  # Softmax with stability
        probs /= probs.sum()
        
        chosen_idx = np.random.choice(len(hold_ids), p=probs)
        chosen_id = hold_ids[chosen_idx]
        
        return chosen_id, list(self.hold_map[chosen_id])
    
    def generate(self, 
             start_lh: int, 
             start_rh: int, 
             max_moves: int = 10,
             temperature: float = 0.01,
             force_alternating: bool = True) -> list[tuple[int, int]]:
        """Generate a climb sequence from starting hold indices."""
        assert temperature > 0

        lh_id, rh_id = start_lh, start_rh
        lh, rh = self.hold_map[start_lh], self.hold_map[start_rh]
        sequence = []
        if self.model.is_sequential:
            hidden = self.model.init_hidden(batch_size=1, device=self.device)
        
        with torch.no_grad():
            for _ in range(max_moves):
                sequence.append((lh_id, rh_id))

                inputs = self._to_input(lh, rh)
                if self.model.is_sequential:
                    predicted, hidden = self.model.forward_step(inputs, hidden)
                else:
                    predicted = self.model.forward(inputs)
                predicted = predicted.cpu().numpy().flatten()
                
                # Sample the hold
                new_lh_id, new_lh = self._sample_hold(predicted[:self.model.num_features],temperature)
                new_rh_id, new_rh = self._sample_hold(predicted[self.model.num_features:],temperature)
                
                # Termination: both hands predict off-wall
                if new_lh == self.model.null_features and new_rh == self.model.null_features:
                    break
                
                if force_alternating:
                    # Calculate movement distance for each hand
                    lh_moved = np.linalg.norm(np.array(lh[:self.model.num_features]) - np.array(new_lh[:self.model.num_features]))
                    rh_moved = np.linalg.norm(np.array(rh[:self.model.num_features]) - np.array(new_rh[:self.model.num_features]))
                    
                    # Only move the hand that moved MORE; snap the other back
                    if lh_moved > rh_moved:
                        lh_id, lh = new_lh_id, new_lh  # Move LH
                        # RH stays as-is
                    else:
                        rh_id, rh = new_rh_id, new_rh  # Move RH
                        # LH stays as-is
                
                # Termination: stuck in same position
                if (lh_id, rh_id) == sequence[-1]:
                    break       
        sequence.append((lh_id, rh_id))
        return sequence