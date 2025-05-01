#!/usr/bin/env python3
# Fixed demo script for hyperparameter tuning with fewer trials and epochs
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from tqdm import tqdm
import argparse

# Define model classes directly in this file to avoid dimension mismatch issues
# Encoder: 1D Convolution Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: [batch_size, n_cities, input_dim]
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, n_cities]
        enc = self.conv(x)      # [batch_size, hidden_dim, n_cities]
        enc = enc.permute(0, 2, 1)  # [batch_size, n_cities, hidden_dim]
        return enc

# Decoder with Attention and GRU
class Decoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        batch_size, n, _ = encoder_outputs.shape
        # Create a new mask for each forward pass - avoid in-place operations
        mask = torch.zeros(batch_size, n, device=encoder_outputs.device).bool()
        d_t = torch.zeros(batch_size, self.hidden_dim, device=encoder_outputs.device)

        tours = []
        log_probs = []

        for i in range(n):
            # Attention mechanism
            e_proj = self.W1(encoder_outputs)  # [batch, n, hidden_dim]
            d_proj = self.W2(d_t).unsqueeze(1)  # [batch, 1, hidden_dim]
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
    def __init__(self, input_dim=4, hidden_dim=128):
        super(Critic, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        enc = self.encoder(x).permute(0, 2, 1)  # [batch, hidden_dim, n]
        value = self.head(enc).squeeze(-1).squeeze(-1)  # [batch]
        return value

# DRL-MOA model combining Actor (Pointer Network) and Critic
class DRLMOA(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super(DRLMOA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim)
        self.critic = Critic(input_dim, hidden_dim)
    
    def forward(self, x):
        encoder_out = self.encoder(x)
        tour, log_probs = self.decoder(encoder_out)
        value = self.critic(x)
        return tour, log_probs, value

# Helper functions
def generate_tsp_data(batch_size, n_cities, input_dim):
    """Generate synthetic Multi-objective TSP instances"""
    # For simplicity, we generate random coordinates and attributes
    # In a real implementation, this would be more sophisticated
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coords = torch.rand(batch_size, n_cities, 2)  # 2D coordinates
    
    # Additional attributes for multi-objective aspects (e.g., cost factors)
    attributes = torch.rand(batch_size, n_cities, input_dim - 2)
    
    # Combine coordinates and attributes
    data = torch.cat([coords, attributes], dim=2)
    return data.to(device)

def compute_reward(tour, coords, weight_vector=None):
    """Compute the scalarized reward for a tour."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if weight_vector is None:
        weight_vector = torch.tensor([0.5, 0.5]).to(device)  # Default equal weights
    
    batch_size = tour.size(0)
    n_cities = tour.size(1)
    rewards = torch.zeros(batch_size).to(device)
    
    for b in range(batch_size):
        # Extract city coordinates for this batch
        cities = coords[b, :, :2]  # Just use the first 2 dimensions (x,y)
        
        # Calculate objectives
        obj1 = 0.0
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
            
            # Calculate a different metric for objective 2
            cost_factor = 1.0
            if coords.size(2) > 2:  # Check if we have additional attributes
                cost_factor = 1.0 + 0.5 * coords[b, from_idx, 2]  # Use attribute as cost factor
            obj2 += dist * cost_factor
        
        # Scalarize objectives
        rewards[b] = -(weight_vector[0] * obj1 + weight_vector[1] * obj2)
    
    return rewards

# Evaluation function
def evaluate_model(model, n_instances=3, n_cities=20, input_dim=4):
    """Evaluate model performance on test instances"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_reward = 0.0
    
    with torch.no_grad():
        for _ in range(n_instances):
            test_data = generate_tsp_data(1, n_cities, input_dim)
            tour, _, _ = model(test_data)
            weight_vector = torch.tensor([0.5, 0.5]).to(device)
            reward = compute_reward(tour, test_data, weight_vector).item()
            total_reward += reward
    
    return total_reward / n_instances

# Define the objective function for Optuna to optimize
def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define hyperparameters to tune
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3])
    
    # Fixed parameters
    input_dim = 4
    n_cities = 20  # Small problem size for quick demo
    num_epochs = 10  # Very few epochs for demo
    
    print(f"Trial with hidden_dim={hidden_dim}, batch_size={batch_size}, lr={learning_rate}")
    
    # Create model with the suggested hyperparameters
    model = DRLMOA(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    model.train()
    try:
        for epoch in range(num_epochs):
            # Generate training data
            coords = generate_tsp_data(batch_size, n_cities, input_dim)
            
            # Forward pass
            tour, log_probs, baseline = model(coords)
            
            # Default weight vector (equal weights)
            weight_vector = torch.tensor([0.5, 0.5]).to(device)
            
            # Compute reward
            reward = compute_reward(tour, coords, weight_vector)
            
            # Compute advantage
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
            
            # Print progress every few epochs
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Reward: {reward.mean().item():.3f}")
    except Exception as e:
        print(f"Error during training: {e}")
        return float('-inf')
    
    # Final evaluation
    val_reward = evaluate_model(model, n_instances=3, n_cities=n_cities, input_dim=input_dim)
    print(f"  Evaluation reward: {val_reward:.3f}")
    return val_reward

def main():
    parser = argparse.ArgumentParser(description='Quick demo of hyperparameter tuning for DRL-MOA')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials for hyperparameter search (quick demo)')
    parser.add_argument('--study_name', type=str, default='drl_moa_tuning_demo', help='Name for the Optuna study')
    parser.add_argument('--output_model', type=str, default='models/drl_moa_tuned_demo.pkl', help='Path to save the tuned model')
    parser.add_argument('--output_params', type=str, default='models/best_params_demo.pkl', help='Path to save the best parameters')
    args = parser.parse_args()

    print("Starting hyperparameter tuning demo")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize', study_name=args.study_name)  # We want to maximize the reward
    study.optimize(objective, n_trials=args.n_trials)  # Just 3 trials for the demo
    
    # Print the best parameters
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Train a final model with the best hyperparameters for a few more epochs
    print("\nTraining final model with best hyperparameters...")
    
    # Create model with the best hyperparameters
    input_dim = 4
    n_cities = 20
    hidden_dim = best_trial.params['hidden_dim']
    learning_rate = best_trial.params['learning_rate']
    batch_size = best_trial.params['batch_size']
    num_epochs = 20  # Train the final model for a few more epochs
    
    best_model = DRLMOA(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=learning_rate)
    
    # Training loop
    best_model.train()
    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        # Generate training data
        coords = generate_tsp_data(batch_size, n_cities, input_dim)
        
        # Forward pass
        tour, log_probs, baseline = best_model(coords)
        
        # Default weight vector (equal weights)
        weight_vector = torch.tensor([0.5, 0.5]).to(device)
        
        # Compute reward
        reward = compute_reward(tour, coords, weight_vector)
        
        # Compute advantage
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
    
    # Save the best model
    torch.save(best_model.state_dict(), args.output_model)
    print(f"Best model state_dict saved to {args.output_model}")
    
    # Also save as .pkl using pickle
    try:
        # Move model to CPU before saving with pickle
        best_model_cpu = best_model.cpu()
        with open(args.output_model, 'wb') as f:
            pickle.dump(best_model_cpu, f)
        print(f"Best model saved to {args.output_model}")
    except Exception as e:
        print(f"Error saving best model with pickle: {e}")
    
    # Create a dictionary with the best parameters and save it
    best_params = {
        'hidden_dim': hidden_dim,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'input_dim': input_dim,
        'n_cities': n_cities,
        'num_epochs': num_epochs
    }
    
    with open(args.output_params, 'wb') as f:
        pickle.dump(best_params, f)
    print(f"Best parameters saved to {args.output_params}")
    
    print("Demo completed!")

if __name__ == "__main__":
    main() 