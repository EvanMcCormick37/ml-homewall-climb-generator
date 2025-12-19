import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple
from app.config import settings
from app.schemas import ModelCreate, ModelType
from app.services.utils.train_data_utils import get_feature_dim


# --- Models ---
class ClimbMLP(nn.Module):
    """Simple multilayer perceptron"""
    def __init__(self,
                 input_dim: int = settings.NUM_LIMBS * settings.NUM_FEATURES,
                 hidden_dim: int = settings.HIDDEN_DIM,
                 output_dim = settings.NUM_LIMBS * settings.NUM_FEATURES,
                 n_hidden_layers: int = settings.N_HIDDEN_LAYERS,
                 ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            )
        for _ in range(n_hidden_layers):
            self.net.append(nn.Linear(hidden_dim, hidden_dim))
            self.net.append(nn.ReLU())
        
        self.net.append(
            nn.Linear(hidden_dim, output_dim)
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

class ClimbRNN(nn.Module):
    """
    Vanilla RNN for climb sequence prediction.
    
    Compatible with existing ClimbGenerator and training loop via forward().
    Use forward_sequence() for full sequence processing with hidden states.
    """
    
    def __init__(self, 
                 output_dim = settings.NUM_LIMBS * settings.NUM_FEATURES,
                 input_dim: int = settings.NUM_LIMBS * settings.NUM_FEATURES,
                 hidden_dim: int = settings.HIDDEN_DIM,
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
                 output_dim = settings.NUM_LIMBS * settings.NUM_FEATURES,
                 input_dim: int = settings.NUM_LIMBS * settings.NUM_FEATURES,
                 hidden_dim: int = settings.HIDDEN_DIM,
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

def create_model_instance(
    config: ModelCreate
    ) -> nn.Module:
    """Create a fresh (untrained) model according to specifications"""
    num_features = get_feature_dim(config.features)
    input_dim = num_features * settings.NUM_LIMBS
    output_dim = num_features * settings.NUM_LIMBS
    if config.model_type == ModelType.MLP:
        return ClimbMLP(
            input_dim,
            settings.HIDDEN_DIM,
            output_dim,
            settings.N_HIDDEN_LAYERS
        ), False
    elif config.model_type == ModelType.RNN:
        return ClimbRNN(
            input_dim,
            settings.HIDDEN_DIM,
            output_dim,
            settings.N_HIDDEN_LAYERS
        ), True
    elif config.model_type == ModelType.LSTM:
        return ClimbLSTM(
            input_dim,
            settings.HIDDEN_DIM,
            output_dim,
            settings.N_HIDDEN_LAYERS
        ), True
    raise TypeError(f"Invalid model type: {config.model_type}")


# --- Training ---
def collate_sequences(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences to the same length within a batch."""
    inputs, targets = zip(*batch)
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0.0)
    return inputs_padded, targets_padded

def run_epoch_mlp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
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

def run_epoch(
    model: nn.Module,
    is_sequential: bool,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: str
) -> Tuple[float,float]:
    if is_sequential:
        return run_epoch_sequential(model,loader,criterion,optimizer,device)
    else:
        return run_epoch_mlp(model,loader,criterion,optimizer,device)