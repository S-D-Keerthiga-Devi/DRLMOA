#!/usr/bin/env python3
# Fixed script to test the trained DRL-MOA model
import torch
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import numpy as np
from model import DRLMOA

def test_model(model_path, n_cities=20, input_dim=4, hidden_dim=128):
    """
    Test a trained DRL-MOA model on a random MOTSP instance
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model from the specified path
    try:
        # First try to load with pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path} using pickle")
    except:
        # If pickle fails, try loading from .pt file
        try:
            # If the file doesn't end with .pt, append it
            if not model_path.endswith('.pt'):
                pt_path = model_path + '.pt'
            else:
                pt_path = model_path
                
            # Create a new model with the same architecture
            model = DRLMOA(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
            
            # Load the state dict
            model.load_state_dict(torch.load(pt_path, map_location=device))
            print(f"Model loaded successfully from {pt_path} using state_dict")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    # Make sure the model is in evaluation mode
    model.eval()
    
    # Create a random TSP instance
    coords = torch.rand(1, n_cities, input_dim).to(device)
    
    # Generate a tour
    with torch.no_grad():
        tour, _, _ = model(coords)
    
    # Convert tour to list of indices
    tour = tour[0].cpu().numpy()
    
    # Extract city coordinates
    city_coords = coords[0, :, :2].cpu().numpy()
    
    # Plot the TSP tour
    plt.figure(figsize=(10, 10))
    plt.scatter(city_coords[:, 0], city_coords[:, 1], s=100, c='blue')
    
    # Add city indices
    for i, (x, y) in enumerate(city_coords):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Connect cities in the tour order
    for i in range(n_cities):
        from_idx = tour[i]
        to_idx = tour[(i + 1) % n_cities]  # Loop back to the first city
        from_x, from_y = city_coords[from_idx]
        to_x, to_y = city_coords[to_idx]
        plt.plot([from_x, to_x], [from_y, to_y], 'r-', alpha=0.7)
    
    # Highlight the start city
    start_idx = tour[0]
    plt.scatter(city_coords[start_idx, 0], city_coords[start_idx, 1], s=200, c='green', marker='*', edgecolors='black', zorder=10)
    
    plt.title(f'TSP Tour (n={n_cities})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(alpha=0.3)
    
    # Save the figure
    plt.savefig('tsp_tour.png')
    print(f"Tour visualization saved to tsp_tour.png")
    
    # Optionally, display the figure
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test a trained DRL-MOA model on a random MOTSP instance')
    parser.add_argument('--model_path', type=str, default='models/drl_moa_tuned_demo.pkl', help='Path to the saved model')
    parser.add_argument('--n_cities', type=int, default=20, help='Number of cities')
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    
    args = parser.parse_args()
    
    test_model(args.model_path, args.n_cities, args.input_dim, args.hidden_dim)

if __name__ == "__main__":
    main() 