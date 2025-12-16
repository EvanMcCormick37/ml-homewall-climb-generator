import torch
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json


class ClimbPositionDataset(Dataset):
    """
    Dataset that extracts (position_t -> position_t+1) pairs from climb sequences.
    """
    def __init__(self, climbs_json_path: str = "data/spraywall-climbs.json", 
                 climb_indices: List[int] = None, 
                 null_hold_idx: int = 263):
        """
        Args:
            climbs_json_path: Path to the spraywall-climbs.json file
            climb_indices: List of climb indices to include in this dataset
            null_hold_idx: Index to use for null holds
        """
        self.examples = []
        self.null_hold_idx = null_hold_idx
        
        # Load climb data
        with open(climbs_json_path, 'r') as f:
            data = json.load(f)
        
        # Extract (input, target) pairs from specified climbs
        for idx in climb_indices:
            climb = data['climbs'][idx]
            
            # Get sequences of hold_ids for this climb
            hold_id_sequence = []
            feature_sequence = []
            
            for position in climb['sequence']:
                position_hold_ids = []
                position_features = []
                
                for hold_data in position['holdsByLimb']:
                    if hold_data == -1:
                        position_hold_ids.append(null_hold_idx)
                        position_features.extend([-1, -1, -1, -1, -1, -1])
                    else:
                        position_hold_ids.append(hold_data['hold_id'])
                        position_features.extend([
                            hold_data['norm_x'],
                            hold_data['norm_y'],
                            hold_data['pull_x'],
                            hold_data['pull_y'],
                            hold_data['useability'] / 10.0,
                            1 if hold_data['type'] == 'hold' else 0
                        ])
                
                hold_id_sequence.append(position_hold_ids)
                feature_sequence.append(np.array(position_features, dtype=np.float32))
            
            # Create training pairs: (features_t -> hold_ids_t+1)
            for t in range(len(feature_sequence) - 1):
                self.examples.append((
                    feature_sequence[t],
                    np.array(hold_id_sequence[t + 1], dtype=np.int64)
                ))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_features, target_hold_ids = self.examples[idx]
        return (
            torch.FloatTensor(input_features),
            torch.LongTensor(target_hold_ids)
        )


class ClimbMLP(nn.Module):
    """
    MLP with shared trunk and 4 independent limb prediction heads.
    """
    def __init__(self, input_dim=24, hidden_dims=[256, 256, 256], num_holds=264):
        super(ClimbMLP, self).__init__()
        
        # Shared trunk: 3 Dense + ReLU layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        # 4 independent output heads (one per limb)
        self.lh_head = nn.Linear(prev_dim, num_holds)
        self.rh_head = nn.Linear(prev_dim, num_holds)
        self.lf_head = nn.Linear(prev_dim, num_holds)
        self.rf_head = nn.Linear(prev_dim, num_holds)
    
    def forward(self, x):
        """
        Args:
            x: Batch of positions [batch_size, 24]
        
        Returns:
            Tuple of 4 tensors, each [batch_size, num_holds] logits
        """
        shared_features = self.trunk(x)
        
        lh_logits = self.lh_head(shared_features)
        rh_logits = self.rh_head(shared_features)
        lf_logits = self.lf_head(shared_features)
        rf_logits = self.rf_head(shared_features)
        
        return lh_logits, rh_logits, lf_logits, rf_logits


def extract_hold_ids_from_holds(holds_json_path: str = "data/holds_final.json") -> Tuple[dict, int]:
    """
    Extract hold_id -> features mapping directly from holds JSON.
    
    Args:
        holds_json_path: Path to holds.json file (default: "data/holds_final.json")
    
    Returns:
        - hold_features_map: dict mapping hold_id -> [6] feature vector
        - null_hold_idx: the index to use for null holds (max_hold_id + 1)
    """
    with open(holds_json_path, 'r') as f:
        data = json.load(f)
    
    hold_features_map = {}
    max_hold_id = -1
    
    # Extract features for each hold
    for hold_data in data['holds']:
        hold_id = hold_data['hold_id']
        max_hold_id = max(max_hold_id, hold_id)
        
        features = np.array([
            hold_data['norm_x'],
            hold_data['norm_y'],
            hold_data['pull_x'],
            hold_data['pull_y'],
            hold_data['useability'] / 10.0,
            1 if hold_data['type'] == 'hold' else 0
        ], dtype=np.float32)
        
        hold_features_map[hold_id] = features
    
    # Null hold gets the next available index
    null_hold_idx = max_hold_id + 1
    hold_features_map[null_hold_idx] = np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32)
    
    print(f"Extracted {len(hold_features_map) - 1} holds (hold_ids: 0-{max_hold_id})")
    print(f"Null hold index: {null_hold_idx}")
    
    return hold_features_map, null_hold_idx


def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cpu'):
    """
    Train the MLP model.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    pbar = tqdm(range(num_epochs), desc="Training")
    
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0.0
        train_correct = [0, 0, 0, 0]  # Accuracy per limb
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            lh_out, rh_out, lf_out, rf_out = model(inputs)
            
            # Calculate loss for each limb
            loss = (
                criterion(lh_out, targets[:, 0]) +
                criterion(rh_out, targets[:, 1]) +
                criterion(lf_out, targets[:, 2]) +
                criterion(rf_out, targets[:, 3])
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            preds = [
                lh_out.argmax(dim=1),
                rh_out.argmax(dim=1),
                lf_out.argmax(dim=1),
                rf_out.argmax(dim=1)
            ]
            for i in range(4):
                train_correct[i] += (preds[i] == targets[:, i]).sum().item()
            train_total += len(inputs)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = [0, 0, 0, 0]
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                lh_out, rh_out, lf_out, rf_out = model(inputs)
                
                loss = (
                    criterion(lh_out, targets[:, 0]) +
                    criterion(rh_out, targets[:, 1]) +
                    criterion(lf_out, targets[:, 2]) +
                    criterion(rf_out, targets[:, 3])
                )
                
                val_loss += loss.item()
                
                preds = [
                    lh_out.argmax(dim=1),
                    rh_out.argmax(dim=1),
                    lf_out.argmax(dim=1),
                    rf_out.argmax(dim=1)
                ]
                for i in range(4):
                    val_correct[i] += (preds[i] == targets[:, i]).sum().item()
                val_total += len(inputs)
        
        # Print statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_acc = [c / train_total * 100 for c in train_correct]
        val_acc = [c / val_total * 100 for c in val_correct]
        
        pbar.set_postfix({
            "Train Acc": f"LH={train_acc[0]:.1f}% RH={train_acc[1]:.1f}% LF={train_acc[2]:.1f}% RF={train_acc[3]:.1f}%",
            "Val Acc": f"LH={val_acc[0]:.1f}% RH={val_acc[1]:.1f}% LF={val_acc[2]:.1f}% RF={val_acc[3]:.1f}%"  
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_climb_mlp.pth')
    print(
        f"Train Acc: LH={train_acc[0]:.1f}% RH={train_acc[1]:.1f}% LF={train_acc[2]:.1f}% RF={train_acc[3]:.1f}%",
        f"Val Acc: LH={val_acc[0]:.1f}% RH={val_acc[1]:.1f}% LF={val_acc[2]:.1f}% RF={val_acc[3]:.1f}%", sep='\n')
    return model


class ClimbGenerator:
    """
    Generate climb sequences autoregressively using a trained MLP model.
    """
    def __init__(self, model: ClimbMLP, hold_features_map: Dict[int, np.ndarray], 
                 device='cpu', null_hold_idx: int = 263):
        """
        Args:
            model: Trained ClimbMLP model
            hold_features_map: Dict mapping hold_id -> feature_vector [6]
            device: 'cpu' or 'cuda'
            null_hold_idx: Index used for null holds
        """
        self.model = model.to(device)
        self.model.eval()
        self.hold_features_map = hold_features_map
        self.device = device
        self.null_hold_idx = null_hold_idx
    
    def hold_indices_to_position(self, hold_indices: List[int]) -> np.ndarray:
        """
        Convert 4 hold indices [LH, RH, LF, RF] to position embedding [24].
        
        Args:
            hold_indices: List of 4 hold indices
        
        Returns:
            Position embedding [24]
        """
        position = []
        for hold_idx in hold_indices:
            if hold_idx in self.hold_features_map:
                position.extend(self.hold_features_map[hold_idx])
            else:
                # Default to null hold if not found
                position.extend([-1, -1, -1, -1, -1, -1])
        
        return np.array(position, dtype=np.float32)
    
    def sample_next_position(self, current_position: np.ndarray, temperature: float = 1.0) -> List[int]:
        """
        Sample the next position given the current position.
        
        Args:
            current_position: Current position embedding [24]
            temperature: Sampling temperature (higher = more random)
                        1.0 = sample from model distribution
                        0.0 = greedy (argmax)
        
        Returns:
            List of 4 hold indices for next position
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(current_position).unsqueeze(0).to(self.device)
            
            # Get logits from model
            lh_logits, rh_logits, lf_logits, rf_logits = self.model(x)
            
            # Apply temperature and sample
            next_holds = []
            for logits in [lh_logits, rh_logits, lf_logits, rf_logits]:
                if temperature == 0.0:
                    # Greedy sampling
                    hold_idx = logits.argmax(dim=1).item()
                else:
                    # Temperature-scaled sampling
                    probs = torch.softmax(logits / temperature, dim=1)
                    hold_idx = torch.multinomial(probs, num_samples=1).item()
                
                next_holds.append(hold_idx)
        
        return next_holds
    
    def generate_climb(self, 
                      start_holds: List[int],
                      max_moves: int = 30,
                      temperature: float = 1.0,
                      stop_on_null: bool = True) -> Tuple[List[List[int]], List[np.ndarray]]:
        """
        Generate a complete climb sequence starting from given holds.
        
        Args:
            start_holds: Initial position as 4 hold indices [LH, RH, LF, RF]
            max_moves: Maximum number of moves to generate
            temperature: Sampling temperature
            stop_on_null: If True, stop when all 4 limbs predict null hold
        
        Returns:
            - sequence_indices: List of positions as hold indices [[LH,RH,LF,RF], ...]
            - sequence_embeddings: List of position embeddings [24]
        """
        sequence_indices = [start_holds]
        current_position = self.hold_indices_to_position(start_holds)
        sequence_embeddings = [current_position]
        
        for move_num in range(max_moves):
            # Sample next position
            next_holds = self.sample_next_position(current_position, temperature)
            sequence_indices.append(next_holds)
            
            # Check for stop condition
            if stop_on_null and all(h == self.null_hold_idx for h in next_holds):
                print(f"Generated climb with {len(sequence_indices)} positions (stopped at null)")
                break
            
            # Convert to embedding for next iteration
            current_position = self.hold_indices_to_position(next_holds)
            sequence_embeddings.append(current_position)
        else:
            print(f"Generated climb with {max_moves} positions (reached max_moves)")
        
        return sequence_indices, sequence_embeddings
    
    def generate_multiple_climbs(self,
                                start_holds: List[int],
                                num_climbs: int = 5,
                                temperature: float = 1.0,
                                **kwargs) -> List[Tuple[List[List[int]], List[np.ndarray]]]:
        """
        Generate multiple climbs from the same starting position.
        Useful for exploring different variations.
        """
        climbs = []
        for i in range(num_climbs):
            print(f"\nGenerating climb {i+1}/{num_climbs}...")
            sequence_indices, sequence_embeddings = self.generate_climb(
                start_holds, temperature=temperature, **kwargs
            )
            climbs.append((sequence_indices, sequence_embeddings))
        
        return climbs


# Convenience function for training
def train_climb_model(holds_json_path: str = "data/holds_final.json",
                     climbs_json_path: str = "data/spraywall-climbs.json",
                     hidden_dims: List[int] = [256, 256, 256],
                     num_epochs: int = 100,
                     batch_size: int = 32,
                     lr: float = 0.001,
                     device: str = None):
    """
    Complete training pipeline for the climb generation model.
    
    Args:
        holds_json_path: Path to holds JSON file
        climbs_json_path: Path to climbs JSON file
        hidden_dims: Hidden layer dimensions for MLP
        num_epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda'), auto-detected if None
    
    Returns:
        Trained model and hold features map
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training on: {device}\n")
    
    # Extract hold features
    hold_features_map, null_hold_idx = extract_hold_ids_from_holds(holds_json_path)
    num_holds = null_hold_idx + 1
    
    print(f"Total hold indices: 0-{null_hold_idx} ({num_holds} total)")
    
    # Determine train/val split on climb indices
    with open(climbs_json_path, 'r') as f:
        data = json.load(f)
    
    num_climbs = len(data['climbs'])
    all_indices = list(range(num_climbs))
    
    # 80/20 split
    split_idx = int(0.8 * num_climbs)
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    print(f"Train climbs: {len(train_indices)} | Val climbs: {len(val_indices)}")
    
    # Create datasets
    train_dataset = ClimbPositionDataset(climbs_json_path, train_indices, null_hold_idx)
    val_dataset = ClimbPositionDataset(climbs_json_path, val_indices, null_hold_idx)
    
    print(f"Train examples: {len(train_dataset)} | Val examples: {len(val_dataset)}\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ClimbMLP(input_dim=24, hidden_dims=hidden_dims, num_holds=num_holds)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device)
    
    return model, hold_features_map, null_hold_idx


# Example usage
if __name__ == "__main__":
    # Train the model
    model, hold_features_map, null_hold_idx = train_climb_model(
        holds_json_path="data/holds_final.json",
        climbs_json_path="data/spraywall-climbs.json",
        num_epochs=100
    )
    
    # Generate some climbs
    print("\n" + "="*60)
    print("GENERATING CLIMBS")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = ClimbGenerator(model, hold_features_map, device=device, null_hold_idx=null_hold_idx)
    
    # Example: Use actual hold_ids from your wall
    start_holds = [206, 139, 243, null_hold_idx]  # LH=206, RH=139, LF=243, RF=null
    
    print(f"Generating climb from starting holds: {start_holds}")
    print(f"(Null hold index: {null_hold_idx})")
    
    sequence, embeddings = generator.generate_climb(
        start_holds=start_holds,
        max_moves=20,
        temperature=1.0
    )
    
    print(f"\nGenerated sequence:")
    for i, holds in enumerate(sequence):
        display_holds = [h if h != null_hold_idx else -1 for h in holds]
        print(f"  Move {i}: LH={display_holds[0]:3d} RH={display_holds[1]:3d} "
              f"LF={display_holds[2]:3d} RF={display_holds[3]:3d}")
