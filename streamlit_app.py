import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set page configuration
st.set_page_config(
    page_title="DRL-MOA for MOTSP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Device configuration
device = torch.device('cpu')  # Force CPU for deployment

# Simple Encoder for display only
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super(SimpleEncoder, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        enc = self.conv(x)
        enc = enc.permute(0, 2, 1)
        return enc

# App title and introduction
st.title("DRL-MOA Framework for Multi-objective TSP")

st.markdown("""
This application demonstrates the Deep Reinforcement Learning for Multi-objective Optimization Algorithm (DRL-MOA) 
framework for solving the Multi-objective Travelling Salesman Problem (MOTSP).

### Key Components:
1. **Problem Decomposition**: Uses Weighted Sum approach to decompose the multi-objective problem into scalar subproblems
2. **Modified Pointer Network**: Sequence-to-sequence model with attention mechanism
3. **Parameter Transfer Strategy**: Neighborhood-based parameter transfer for efficient training
4. **Actor-Critic DRL**: Unsupervised training using the Actor-Critic algorithm
5. **Pareto Front Visualization**: Plotting the non-dominated solutions for evaluation
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Model Implementation", "Demo"])

# Introduction page
if page == "Introduction":
    st.header("Introduction to DRL-MOA")
    
    st.subheader("Multi-objective Travelling Salesman Problem (MOTSP)")
    
    st.markdown("""
    The Multi-objective Travelling Salesman Problem extends the classic TSP by considering multiple,
    often conflicting objectives. For example:
    - Minimize total distance traveled
    - Minimize total time
    - Minimize altitude changes
    - Minimize fuel consumption
    
    Unlike single-objective optimization, there is no single optimal solution but rather a set of 
    Pareto-optimal solutions representing different trade-offs between objectives.
    """)
    
    # Display example problem visually
    st.subheader("Example Problem Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Objective 1: Minimize Distance (XY Coordinates)**")
        
        # Generate a sample problem
        n_cities = 10
        np.random.seed(42)
        cities1 = np.random.rand(n_cities, 2)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(cities1[:, 0], cities1[:, 1], s=100, c='blue')
        
        # Add city indices
        for i, (x, y) in enumerate(cities1):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
            
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Cities - Objective 1')
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Objective 2: Minimize Altitude Changes (Different XY Coordinates)**")
        
        # Generate different coordinates for the second objective
        np.random.seed(43)
        cities2 = np.random.rand(n_cities, 2)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(cities2[:, 0], cities2[:, 1], s=100, c='red')
        
        # Add city indices
        for i, (x, y) in enumerate(cities2):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
            
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Cities - Objective 2')
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)

# Model Implementation page
elif page == "Model Implementation":
    st.header("DRL-MOA Model Implementation")
    
    st.markdown("""
    This page presents the implementation details of our Deep Reinforcement Learning for Multi-Objective Optimization Algorithm (DRL-MOA)
    for solving the Multi-objective Traveling Salesman Problem (MOTSP).
    
    The model is implemented in PyTorch and consists of several key components that work together to solve this complex optimization problem.
    """)
    
    # Model architecture description
    st.subheader("Model Architecture")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        ### Deep Neural Network Components
        
        The DRL-MOA architecture consists of several neural network modules:
        
        1. **Encoder**: Transforms city coordinates into embeddings
           - 1D Convolutional network
           - Projects input features into hidden dimension
        
        2. **Decoder**: Builds tours one city at a time
           - Attention mechanism
           - GRU cell for state tracking
           - Outputs probabilities over cities
        
        3. **Critic**: Estimates expected rewards
           - Value network
           - Used for advantage calculation
        
        4. **Actor-Critic Framework**: Combines all components
           - Policy gradient loss
           - Value function loss
        """)
    
    with col2:
        st.image("https://raw.githubusercontent.com/S-D-Keerthiga-Devi/DRLMOA/main/comparison_plots/cost_comparison.png", 
                caption="Performance Comparison")

# Demo page
elif page == "Demo":
    st.header("Interactive Demo")
    
    st.subheader("Generate a Random TSP Instance")
    
    n_cities = st.slider("Number of Cities", min_value=5, max_value=30, value=10, step=1)
    
    if st.button("Generate New Instance"):
        # Generate a random problem
        np.random.seed(np.random.randint(1000))
        cities = np.random.rand(n_cities, 2)
        
        # Create a simple tour (just for demonstration)
        tour = list(range(n_cities))
        np.random.shuffle(tour)
        tour.append(tour[0])  # Complete the tour
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(cities[:, 0], cities[:, 1], s=100, c='blue')
        
        # Add city indices
        for i, (x, y) in enumerate(cities):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plot the tour
        for i in range(len(tour)-1):
            from_idx = tour[i]
            to_idx = tour[i+1]
            ax.plot([cities[from_idx, 0], cities[to_idx, 0]], 
                    [cities[from_idx, 1], cities[to_idx, 1]], 'r-', alpha=0.7)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Random TSP Tour ({n_cities} cities)')
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)
        
        # Calculate "objectives" (for demo purposes)
        dist = 0
        for i in range(len(tour)-1):
            from_idx = tour[i]
            to_idx = tour[i+1]
            dist += np.sqrt(((cities[from_idx] - cities[to_idx])**2).sum())
        
        # Show metrics
        st.metric("Tour Length", f"{dist:.2f}")
        
    st.markdown("""
    ### About the Algorithm
    
    The full implementation of DRL-MOA applies deep reinforcement learning to find near-optimal 
    solutions to multi-objective TSP problems. In a real deployment, the model would:
    
    1. Encode the city coordinates
    2. Use a pointer network to generate tours
    3. Evaluate multiple objectives
    4. Learn better policies through reinforcement learning
    
    This demo provides a simplified visualization of the problem and solution approach.
    """)

# Footer
st.markdown("---")
st.markdown("DRL-MOA Project for Multi-objective TSP | [GitHub Repository](https://github.com/S-D-Keerthiga-Devi/DRLMOA)") 