#!/usr/bin/env python3
# Hyperparameter tuning for the DRL-MOA model using Optuna
import optuna
import torch
import numpy as np
import argparse
import pickle
import os
from tqdm import tqdm
from model_fixed import DRLMOA, generate_tsp_data, compute_reward, device

# Evaluation function to test model performance
def evaluate_model(model, n_instances=5, n_cities=40, input_dim=4):
    """Evaluate model performance on test instances"""
    model.eval()
    total_reward = 0.0
    
    with torch.no_grad():
        for _ in range(n_instances):
            # Generate test instance
            test_data = generate_tsp_data(1, n_cities, input_dim)
            
            # Run inference
            tour, _, _ = model(test_data)
            
            # Compute reward (use equal weights for evaluation)
            weight_vector = torch.tensor([0.5, 0.5]).to(device)
            reward = compute_reward(tour, test_data, weight_vector).item()
            total_reward += reward
    
    # Return average reward (higher is better)
    return total_reward / n_instances

# Define the objective function for Optuna to optimize
def objective(trial):
    # Define hyperparameters to tune
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
    batch_size = trial.suggest_int('batch_size', 16, 128, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # Fixed parameters
    input_dim = 4
    n_cities = 30  # Use a smaller problem size for faster tuning
    num_epochs = 50  # Use fewer epochs for faster iterations
    
    # Create model with the suggested hyperparameters
    model = DRLMOA(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    
    # Create optimizer with the suggested learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model (simplified training loop from model_fixed.py)
    model.train()
    
    # Track best validation reward
    best_reward = float('-inf')
    
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
            critic_loss = torch.nn.functional.mse_loss(baseline, reward)
            
            # Total loss
            loss = actor_loss + critic_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Evaluate model periodically
            if (epoch + 1) % 10 == 0:
                val_reward = evaluate_model(model, n_instances=3, n_cities=n_cities, input_dim=input_dim)
                if val_reward > best_reward:
                    best_reward = val_reward
                
                # Report intermediate value to Optuna
                trial.report(val_reward, epoch)
                
                # Handle pruning based on the intermediate value
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
    
    except Exception as e:
        # Handle errors during training
        print(f"Error during training: {e}")
        return float('-inf')
    
    return best_reward

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for DRL-MOA')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials for hyperparameter search')
    parser.add_argument('--study_name', type=str, default='drl_moa_tuning', help='Name for the Optuna study')
    parser.add_argument('--storage', type=str, default=None, help='Storage URL for the study')
    parser.add_argument('--output_model', type=str, default='models/drl_moa_tuned.pkl', help='Path to save the tuned model')
    args = parser.parse_args()
    
    print(f"Starting hyperparameter tuning with {args.n_trials} trials")
    print(f"Device: {device}")
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',  # We want to maximize the reward
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(objective, n_trials=args.n_trials)
    
    # Print the best parameters
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Train the final model with the best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    
    # Create model with the best hyperparameters
    input_dim = 4
    n_cities = 40  # Use the standard problem size for final model
    hidden_dim = best_trial.params['hidden_dim']
    learning_rate = best_trial.params['learning_rate']
    batch_size = best_trial.params['batch_size']
    num_epochs = 200  # Train the final model for more epochs
    
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
        critic_loss = torch.nn.functional.mse_loss(baseline, reward)
        
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
    torch.save(best_model.state_dict(), f"{args.output_model}.pt")
    print(f"Best model state_dict saved to {args.output_model}.pt")
    
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
    
    with open('best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    print("Best parameters saved to best_params.pkl")
    
    print("Hyperparameter tuning completed!")

if __name__ == "__main__":
    main() 