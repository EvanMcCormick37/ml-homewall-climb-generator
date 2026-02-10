import torch
from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from tqdm import tqdm
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------
# DATA BATCHING AND PREPROCESSING
#-----------------------------------------------------------------------
def sort_by_length(batch):
    """Collate function for preparing to Pack-sort the tensors, by sorting them in order from longest to shortest."""
    sorted_batch = sorted(batch, key=lambda b: b[2], reverse=True)
    x, y, l = zip(*sorted_batch)
    return (
        torch.stack(x),
        torch.stack(y),
        torch.tensor(l)
    )

def train_lstm_on_packed_dataset(model: nn.Module, dataset: TensorDataset, epochs: int = 100, lr=1e-3, batch_size=256, save_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    model.compile()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sort_by_length,
        pin_memory=True if device.type == 'cuda' else False
    )
    losses = []

    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            epoch_losses = []
            for (idx, (x, y, lengths)) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                packed_input = pack_padded_sequence(
                    input = x,
                    lengths = lengths,
                    batch_first = True,
                    enforce_sorted = True
                )

                # Model prediction and loss update
                optimizer.zero_grad()
                pred_logits = model(packed_input)
                loss = model.loss(pred_logits, y)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
            mean_loss = sum(epoch_losses)/len(epoch_losses)
            losses.append(mean_loss)
            pbar.set_postfix_str(f"Epoch: {epoch}, Mean Loss: {mean_loss}, Min Loss: {min(losses)}, {len(epoch_losses)} Batches")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
    
    fig, ax = plt.subplots()

    ax.plot(list(range(len(losses))), losses)
    ax.set_yscale('log')
    ax.set_title(f'Mean Batch-Loss per Epoch, LSTM Hold Classifier ({sum([p.numel() for p in model.parameters()])} params)')
    plt.show()

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