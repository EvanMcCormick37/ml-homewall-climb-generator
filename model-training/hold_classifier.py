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
@DeprecationWarning
def sort_by_length(batch):
    """Collate function for preparing to Pack-sort the tensors, by sorting them in order from longest to shortest."""
    sorted_batch = sorted(batch, key=lambda b: b[2], reverse=True)
    x, y, l = zip(*sorted_batch)
    return (
        torch.stack(x),
        torch.stack(y),
        torch.tensor(l)
    )

@DeprecationWarning
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

@DeprecationWarning
class HoldClassifierLSTM(nn.Module):
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

    
    def forward(self, holds_cond: PackedSequence | Tensor)-> Tensor:
        """Run the forward pass. Predicts the roles for a given (possibly batched) set of holds, given (possibly batched) wall conditions."""

        _, (hs, cs) = self.lstm(holds_cond)

        lstm_final_state = torch.cat([hs[-1],hs[-2]], dim=1)
        
        sf_logits = self.classification_head(lstm_final_state)

        return sf_logits

class ResidualBlock1D_V2(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        self.cond_proj = nn.Linear(cond_dim, out_channels*2)
        self.shortcut = nn.Conv1d(in_channels,out_channels,1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.norm1(h)

        gamma, beta = self.cond_proj(cond).unsqueeze(-1).chunk(2, dim=1)
        h = h*(1+gamma) + beta
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.shortcut(x)

class UNetHoldClassifierLogits(nn.Module):
    def __init__(
        self,
        in_features_dim: int = 4,
        in_cond_dim: int = 4,
        out_dim: int = 5,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cond_emb = nn.Sequential(
            nn.Linear(in_cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.init_conv = ResidualBlock1D_V2(in_features_dim, hidden_dim, hidden_dim)

        self.down_blocks = nn.ModuleList([ResidualBlock1D_V2(hidden_dim*(i+1), hidden_dim*(i+2), hidden_dim) for i in range(n_layers)])
        self.up_blocks = nn.ModuleList([ResidualBlock1D_V2(hidden_dim*(i+1), hidden_dim*(i), hidden_dim) for i in range(n_layers,0,-1)])

        self.head = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x, cond):

        x = zero_com(x, 2)

        cond_emb = self.cond_emb(cond)
        h_emb = self.init_conv(x.transpose(1,2), cond_emb)

        residuals = []
        for layer in self.down_blocks:
            residuals.append(h_emb)
            h_emb = layer(h_emb, cond_emb)
        
        for layer in self.up_blocks:
            resid = residuals.pop()
            h_emb = resid + layer(h_emb, cond_emb)
        
        h_out = self.head(h_emb).transpose(1,2)

        return h_out

def train_unet_hold_classifier_logits(
    hold_classifier: nn.Module,
    dataset: TensorDataset,
    epochs: int = 100,
    batch_size: int = 128,
    num_workers: int = 0,
    save_path: str | None = None,
    save_on_best: bool = True,
    torch_compile: bool = False
) -> tuple[nn.Module, list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Set-Up
    hold_classifier.to(device)
    hold_classifier.train()
    if torch_compile:
        hold_classifier = torch.compile(hold_classifier)
    optimizer = torch.optim.Adam(params = hold_classifier.parameters())

    n_params = sum([p.numel() for p in hold_classifier.parameters()])

    # DataLoader Set-Up
    batches = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    
    epoch_losses = []
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            batch_losses = []
            for x, cond, target_role_probs in batches:
                x, cond = x.to(device), cond.to(device)
                optimizer.zero_grad()

                pred_logits = hold_classifier(x, cond)

                loss = F.cross_entropy(pred_logits.transpose(1,2), torch.argmax(target_role_probs,dim=2))

                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
            mean_batch_loss = sum(batch_losses)/len(batch_losses)
            epoch_losses.append(mean_batch_loss)
            info_str = f"Epoch: {epoch}: Avg Batch Loss: {mean_batch_loss:.3f}, Min Batch Loss: {min(epoch_losses):.3f}. {len(batch_losses)} batches, {n_params} params"
            if save_path and save_on_best and (min(epoch_losses)==mean_batch_loss):
                pbar.set_postfix_str(f"{info_str}. New best mean batch loss! Saving hold classifier at {save_path}...")
                torch.save(hold_classifier.state_dict(), save_path)
            else:
                pbar.set_postfix_str(info_str)

    if save_path:
        print(f"Saving hold classifier at {save_path}...")
        torch.save(hold_classifier.state_dict(), save_path)
    
    
    # Plot training results
    fig, ax = plt.subplots()

    ax.plot(list(range(len(epoch_losses))), epoch_losses)
    ax.set_yscale('log')
    ax.set_title(f'Mean Batch-Loss per Epoch, U-Net Hold Classifier ({n_params} params)')
    plt.show()

    return hold_classifier, epoch_losses
