#!/usr/bin/env python3
# Script to test and compare the tuned model with the original model
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import os
from tuning_demo_fixed import DRLMOA, generate_tsp_data, compute_reward

def load_model(model_path, input_dim=4, hidden_dim=128):
    """Load a trained model from a .pkl file or .pt file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try loading with pickle first
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from {model_path}")
        return model.to(device)
    except Exception as e:
        print(f"Error loading model with pickle: {e}")
        
        # Try loading the model state dict
        try:
            # If .pkl file failed, try .pt file
            pt_path = model_path + ".pt" if not model_path.endswith(".pt") else model_path
            
            # Create a new model instance
            model = DRLMOA(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
            
            # Load the state dict
            model.load_state_dict(torch.load(pt_path, map_location=device))
            print(f"Successfully loaded model state_dict from {pt_path}")
            return model
        except Exception as e2:
            print(f"Error loading model state_dict: {e2}")
            return None

def generate_test_instances(n_instances, n_cities, input_dim=4):
    """Generate multiple test instances with fixed random seed for fair comparison"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)  # Fixed seed for reproducibility
    test_instances = []
    for _ in range(n_instances):
        coords = torch.rand(1, n_cities, 2)  # 2D coordinates
        attributes = torch.rand(1, n_cities, input_dim - 2)
        data = torch.cat([coords, attributes], dim=2)
        test_instances.append(data.to(device))
    return test_instances

def evaluate_model_on_instances(model, test_instances):
    """Evaluate a model on a set of test instances"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    rewards = []
    tours = []
    
    with torch.no_grad():
        for test_data in test_instances:
            # Run inference
            tour, _, _ = model(test_data)
            
            # Get tour (cities sequence)
            tour_numpy = tour[0].cpu().numpy()
            
            # Compute reward
            tour_tensor = torch.tensor(tour_numpy).unsqueeze(0).to(device)
            weight_vector = torch.tensor([0.5, 0.5]).to(device)
            reward = compute_reward(tour_tensor, test_data, weight_vector).item()
            
            rewards.append(-reward)  # Convert back to cost (negative reward)
            tours.append(tour_numpy)
    
    return rewards, tours

def plot_tour_comparison(original_tour, tuned_tour, cities, original_cost, tuned_cost, instance_idx, n_cities):
    """Plot two tours side by side for comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Extract coordinates
    x = cities[:, 0]
    y = cities[:, 1]
    
    # Plot original tour
    ax1.scatter(x, y, s=100, c='blue', zorder=10)
    for i in range(len(original_tour)):
        from_idx = original_tour[i]
        to_idx = original_tour[(i + 1) % len(original_tour)]
        ax1.plot([x[from_idx], x[to_idx]], [y[from_idx], y[to_idx]], 'r-', alpha=0.7)
    ax1.set_title(f'Original Model Tour\nCost: {original_cost:.4f}')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.grid(True)
    
    # Plot tuned tour
    ax2.scatter(x, y, s=100, c='blue', zorder=10)
    for i in range(len(tuned_tour)):
        from_idx = tuned_tour[i]
        to_idx = tuned_tour[(i + 1) % len(tuned_tour)]
        ax2.plot([x[from_idx], x[to_idx]], [y[from_idx], y[to_idx]], 'g-', alpha=0.7)
    ax2.set_title(f'Tuned Model Tour\nCost: {tuned_cost:.4f}')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.grid(True)
    
    plt.suptitle(f'Tour Comparison - Instance {instance_idx+1} ({n_cities} cities)')
    plt.tight_layout()
    
    # Save figure
    output_dir = 'comparison_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/comparison_instance_{instance_idx+1}.png')
    
    return fig

def plot_cost_comparison(original_costs, tuned_costs):
    """Plot a bar chart comparing costs across instances"""
    n_instances = len(original_costs)
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(n_instances)
    width = 0.35
    
    plt.bar(x - width/2, original_costs, width, label='Original Model')
    plt.bar(x + width/2, tuned_costs, width, label='Tuned Model')
    
    plt.xlabel('Test Instance')
    plt.ylabel('Tour Cost (lower is better)')
    plt.title('Cost Comparison: Original vs. Tuned Model')
    plt.xticks(x, [f'Instance {i+1}' for i in range(n_instances)])
    plt.legend()
    plt.grid(True, axis='y')
    
    # Calculate average improvement
    avg_improvement = np.mean([(original - tuned) / original * 100 for original, tuned in zip(original_costs, tuned_costs)])
    if avg_improvement > 0:
        plt.figtext(0.5, 0.01, f'Average improvement: {avg_improvement:.2f}% reduction in cost', 
                   horizontalalignment='center', fontsize=12)
    else:
        plt.figtext(0.5, 0.01, f'Average change: {-avg_improvement:.2f}% increase in cost', 
                   horizontalalignment='center', fontsize=12)
    
    # Save figure
    output_dir = 'comparison_plots'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/cost_comparison.png')
    
    return plt.gcf()

def main():
    parser = argparse.ArgumentParser(description='Compare original and tuned DRL-MOA models')
    parser.add_argument('--original_model', type=str, default='models/drl_moa_model.pkl', help='Path to the original model')
    parser.add_argument('--tuned_model', type=str, default='models/drl_moa_tuned_demo.pkl', help='Path to the tuned model')
    parser.add_argument('--n_cities', type=int, default=20, help='Number of cities for test instances')
    parser.add_argument('--n_instances', type=int, default=5, help='Number of test instances')
    parser.add_argument('--input_dim', type=int, default=4,
                      help='Input dimension')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the original model
    print(f"Loading original model from {args.original_model}...")
    original_model = load_model(args.original_model, args.input_dim)
    if original_model is None:
        print("Could not load original model. Exiting.")
        return
    
    # Load the tuned model
    print(f"Loading tuned model from {args.tuned_model}...")
    
    # First try to load best parameters
    try:
        with open('best_params_demo.pkl', 'rb') as f:
            best_params = pickle.load(f)
            hidden_dim = best_params.get('hidden_dim', 128)
            print(f"Using hidden_dim={hidden_dim} from best_params_demo.pkl")
    except Exception as e:
        print(f"Error loading best parameters: {e}")
        hidden_dim = 128
    
    tuned_model = load_model(args.tuned_model, args.input_dim, hidden_dim)
    if tuned_model is None:
        print("Could not load tuned model. Exiting.")
        return
    
    # Generate test instances
    print(f"Generating {args.n_instances} test instances with {args.n_cities} cities each...")
    test_instances = generate_test_instances(args.n_instances, args.n_cities, args.input_dim)
    
    # Evaluate original model
    print("Evaluating original model...")
    original_costs, original_tours = evaluate_model_on_instances(original_model, test_instances)
    print(f"Original model average cost: {np.mean(original_costs):.4f}")
    
    # Evaluate tuned model
    print("Evaluating tuned model...")
    tuned_costs, tuned_tours = evaluate_model_on_instances(tuned_model, test_instances)
    print(f"Tuned model average cost: {np.mean(tuned_costs):.4f}")
    
    # Calculate improvement
    improvements = [(original - tuned) / original * 100 for original, tuned in zip(original_costs, tuned_costs)]
    avg_improvement = np.mean(improvements)
    
    print("\nResults:")
    print(f"{'Instance':<10} {'Original Cost':<15} {'Tuned Cost':<15} {'Improvement':<15}")
    print("-" * 55)
    for i, (orig, tuned, imp) in enumerate(zip(original_costs, tuned_costs, improvements)):
        print(f"{i+1:<10} {orig:<15.4f} {tuned:<15.4f} {imp:+<15.2f}%")
    print("-" * 55)
    print(f"{'Average':<10} {np.mean(original_costs):<15.4f} {np.mean(tuned_costs):<15.4f} {avg_improvement:+<15.2f}%")
    
    # Plot tour comparisons
    print("\nGenerating comparison plots...")
    os.makedirs('comparison_plots', exist_ok=True)
    
    for i in range(min(3, args.n_instances)):  # Plot at most 3 instances
        cities = test_instances[i][0, :, :2].cpu().numpy()
        plot_tour_comparison(
            original_tours[i], tuned_tours[i], cities,
            original_costs[i], tuned_costs[i], i, args.n_cities
        )
    
    # Plot overall cost comparison
    plot_cost_comparison(original_costs, tuned_costs)
    
    print(f"Plots saved to comparison_plots/ directory")
    
    # Summary conclusion
    if avg_improvement > 0:
        print(f"\nConclusion: The tuned model performs {avg_improvement:.2f}% better than the original model on average.")
    else:
        print(f"\nConclusion: The tuned model performs {-avg_improvement:.2f}% worse than the original model on average.")
        print("Note: This can happen with limited tuning. Try more trials and training epochs for better results.")

if __name__ == "__main__":
    main() 