import torch
from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

#-----------------------------------------------------------------------
# HOLD ROLE CLASSIFICATION
#-----------------------------------------------------------------------

class HoldClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 256, num_layers: int = 1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout = 0.3,
            bidirectional=True,
            device=self.device
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        self.loss_func = nn.BCEWithLogitsLoss()
    
    def loss(self, pred_roles: Tensor, true_roles: Tensor):
        """Get the loss from the model's predictions, via cross-entropy loss."""
        return self.loss_func(pred_roles, true_roles)

    
    def forward(self, holds_cond: PackedSequence)-> Tensor:
        """Run the forward pass. Predicts the roles for a given (possibly batched) set of holds, given (possibly batched) wall conditions."""

        _, (hs, cs) = self.lstm(holds_cond)

        lstm_final_state = torch.cat([hs[-1],hs[-2]], dim=1)

        sf_logits = self.classification_head(lstm_final_state)

        return sf_logits

