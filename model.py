#!/usr/bin/env python3
# Fixed DRL-MOA Pointer Network Implementation for Multi-objective TSP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm

# Hyperparameters
D_input = 4         # Input dimension (e.g., 2 coords + 2 attributes for bi-objective Euclidean)
dh = 128            # Hidden size
n_cities = 40       # Number of cities
batch_size = 128    # Batch size for training
num_epochs = 500    # Number of training epochs
lr = 1e-4           # Learning rate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder: 1D Convolution Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim=D_input, hidden_dim=dh):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: [batch_size, n_cities, D_input]
        x = x.permute(0, 2, 1)  # [batch_size, D_input, n_cities]
        enc = self.conv(x)      # [batch_size, hidden_dim, n_cities]
        enc = enc.permute(0, 2, 1)  # [batch_size, n_cities, hidden_dim]
        return enc

# Decoder with Attention and GRU - Fixed to avoid inplace operations
class Decoder(nn.Module):
    def __init__(self, hidden_dim=dh):
        super(Decoder, self).__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        batch_size, n, _ = encoder_outputs.shape
        # Create a new mask for each forward pass - avoid in-place operations
        mask = torch.zeros(batch_size, n, device=encoder_outputs.device).bool()
        d_t = torch.zeros(batch_size, dh, device=encoder_outputs.device)

        tours = []
        log_probs = []

        for i in range(n):
            # Attention mechanism
            e_proj = self.W1(encoder_outputs)  # [batch, n, dh]
            d_proj = self.W2(d_t).unsqueeze(1)  # [batch, 1, dh]
            u_t = self.v(torch.tanh(e_proj + d_proj)).squeeze(-1)  # [batch, n]

            # Instead of in-place operation, create a new tensor
            masked_u_t = u_t.clone()
            masked_u_t = masked_u_t.masked_fill(mask, float('-inf'))  
            p_t = F.softmax(masked_u_t, dim=-1)  # Prob distribution over cities

            dist = torch.distributions.Categorical(p_t)
            idx = dist.sample()  # Sample next city

            log_probs.append(dist.log_prob(idx))
            tours.append(idx)

            # Update the hidden state
            idx_gather = idx.unsqueeze(1).expand(-1, encoder_outputs.size(2))
            selected_embeddings = encoder_outputs.gather(1, idx_gather.unsqueeze(1)).squeeze(1)
            d_t = self.gru(selected_embeddings, d_t)
            
            # Update mask (without in-place operations)
            new_mask = mask.clone()
            new_mask.scatter_(1, idx.unsqueeze(-1), True)
            mask = new_mask

        return torch.stack(tours, dim=1), torch.stack(log_probs, dim=1)

# Critic network
class Critic(nn.Module):
    def __init__(self, input_dim=D_input, hidden_dim=dh):
        super(Critic, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        enc = self.encoder(x).permute(0, 2, 1)  # [batch, hidden, n]
        value = self.head(enc).squeeze(-1).squeeze(-1)  # [batch]
        return value

# DRL-MOA model combining Actor (Pointer Network) and Critic
class DRLMOA(nn.Module):
    def __init__(self, input_dim=D_input, hidden_dim=dh):
        super(DRLMOA, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim)
        self.critic = Critic(input_dim, hidden_dim)
    
    def forward(self, x):
        encoder_out = self.encoder(x)
        tour, log_probs = self.decoder(encoder_out)
        value = self.critic(x)
        return tour, log_probs, value

# Generate synthetic data for MOTSP
def generate_tsp_data(batch_size, n_cities, input_dim):
    """Generate synthetic Multi-objective TSP instances"""
    # For simplicity, we generate random coordinates and attributes
    # In a real implementation, this would be more sophisticated
    coords = torch.rand(batch_size, n_cities, 2)  # 2D coordinates
    
    # Additional attributes for multi-objective aspects (e.g., cost factors)
    attributes = torch.rand(batch_size, n_cities, input_dim - 2)
    
    # Combine coordinates and attributes
    data = torch.cat([coords, attributes], dim=2)
    return data.to(device)

# Compute reward for a tour (scalarized for a single subproblem)
def compute_reward(tour, coords, weight_vector=None):
    """
    Compute the scalarized reward for a tour.
    This is a simplified implementation that:
    1. Calculates Euclidean distances between consecutive cities
    2. Uses a different metric for the second objective
    3. Scalarizes using the weight vector
    """
    if weight_vector is None:
        weight_vector = torch.tensor([0.5, 0.5]).to(device)  # Default equal weights
    
    batch_size = tour.size(0)
    n_cities = tour.size(1)
    rewards = torch.zeros(batch_size).to(device)
    
    for b in range(batch_size):
        # Extract city coordinates for this batch
        cities = coords[b, :, :2]  # Just use the first 2 dimensions (x,y)
        
        # Calculate distances for objective 1 (Euclidean distance)
        obj1 = 0.0
        # Calculate costs for objective 2 (could be time, fuel, etc.)
        obj2 = 0.0
        
        # Loop through cities in the tour
        for i in range(n_cities):
            from_idx = tour[b, i]
            to_idx = tour[b, (i + 1) % n_cities]  # Wrap around to first city
            
            # Calculate Euclidean distance for objective 1
            from_city = cities[from_idx]
            to_city = cities[to_idx]
            dist = torch.sqrt(torch.sum((from_city - to_city) ** 2))
            obj1 += dist
            
            # Calculate a different metric for objective 2 (e.g., modified distance)
            # For simplicity, we'll use a random factor multiplied by the distance
            # Make sure index is in bounds for coords
            cost_factor = 1.0
            if coords.size(2) > 2:  # Check if we have additional attributes
                cost_factor = 1.0 + 0.5 * coords[b, from_idx, 2]  # Use attribute as cost factor
            obj2 += dist * cost_factor
        
        # Scalarize the objectives using the weight vector
        # Negative because we want to maximize reward (minimize cost)
        rewards[b] = -(weight_vector[0] * obj1 + weight_vector[1] * obj2)
    
    return rewards

def train_model(model, num_epochs, batch_size, n_cities, input_dim, lr, weight_vector=None, save_path='drl_moa_model.pkl'):
    """Train the DRL-MOA model for a single subproblem"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training with batch size: {batch_size}, cities: {n_cities}, input_dim: {input_dim}")
    
    # Training loop
    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        # Generate synthetic MOTSP instances
        coords = generate_tsp_data(batch_size, n_cities, input_dim)
        
        # Forward pass
        tour, log_probs, baseline = model(coords)
        
        # Compute reward
        reward = compute_reward(tour, coords, weight_vector)
        
        # Compute advantage (reward - baseline)
        advantage = reward - baseline.detach()
        
        # Actor loss (policy gradient with baseline)
        actor_loss = -(advantage * log_probs.sum(dim=1)).mean()
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(baseline, reward)
        
        # Total loss
        loss = actor_loss + critic_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        progress_bar.set_description(
            f"Epoch {epoch+1}/{num_epochs}, Reward: {reward.mean().item():.3f}, Loss: {loss.item():.3f}"
        )
    
    # Save the trained model
    torch.save(model.state_dict(), f"{save_path}.pt")
    print(f"Model state_dict saved to {save_path}.pt")
    
    # Also save as .pkl using pickle
    try:
        # Move model to CPU before saving with pickle
        model_cpu = model.cpu()
        with open(save_path, 'wb') as f:
            pickle.dump(model_cpu, f)
        print(f"Model saved to {save_path}")
        
        # Move model back to original device
        model.to(device)
    except Exception as e:
        print(f"Error saving model with pickle: {e}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train DRL-MOA Pointer Network for MOTSP')
    parser.add_argument('--input_dim', type=int, default=D_input, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=dh, help='Hidden dimension')
    parser.add_argument('--n_cities', type=int, default=n_cities, help='Number of cities')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=lr, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='drl_moa_model.pkl', help='Path to save model')
    args = parser.parse_args()
    
    # Create model
    model = DRLMOA(
        input_dim=args.input_dim, 
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Print model summary
    print(f"Model architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print(f"Training DRL-MOA model on {device}...")
    train_model(
        model=model,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        n_cities=args.n_cities,
        input_dim=args.input_dim,
        lr=args.lr,
        save_path=args.save_path
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main() 