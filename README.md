# DRL-MOA for Multi-Objective TSP

This repository implements a simplified version of the Deep Reinforcement Learning for Multi-Objective Optimization Algorithm (DRL-MOA) applied to the Multi-Objective Traveling Salesman Problem (MOTSP).

## Overview

The implementation includes:

1. **Pointer Network Architecture**: Uses a 1D convolution encoder and attention-based GRU decoder to construct TSP tours.
2. **Critic Network**: Estimates the expected reward for the actor-critic training framework.
3. **Scalarization**: The multi-objective problem is scalarized using weight vectors.
4. **Model Saving**: Trained models are saved to .pkl and .pt files for later use.
5. **Hyperparameter Tuning**: Automated optimization of model hyperparameters using Optuna.

## Files

- `model.py` - Core implementation of the DRL-MOA pointer network model
- `model_fixed.py` - Fixed version that resolves in-place operation errors in backpropagation
- `run.py` - Script to train and save the model
- `test_model.py` - Script to load and test a trained model
- `test_model_fixed.py` - Fixed version that handles loading models correctly
- `hyperparameter_tuning.py` - Script for full hyperparameter optimization using Optuna
- `tuning_demo_fixed.py` - Quick demo of hyperparameter tuning with fewer trials
- `test_tuned_model.py` - Script to compare original and tuned models
- `README.md` - This documentation file
- `app.py` - Streamlit web application for interactive demonstration

## Requirements

- PyTorch
- NumPy
- Matplotlib
- tqdm
- Optuna (for hyperparameter tuning)

## Usage

### Training a Model

To train a model with default parameters:

```bash
python model_fixed.py
```

With custom parameters:

```bash
python model_fixed.py --n_cities 50 --num_epochs 1000 --batch_size 64 --save_path my_model.pkl
```

Parameters:
- `--input_dim` - Input dimension (default: 4)
- `--hidden_dim` - Hidden dimension (default: 128)
- `--n_cities` - Number of cities (default: 40)
- `--batch_size` - Batch size (default: 128)
- `--num_epochs` - Number of training epochs (default: 500)
- `--lr` - Learning rate (default: 1e-4)
- `--save_path` - Path to save model (default: drl_moa_model.pkl)

### Testing a Trained Model

To test a trained model and visualize a TSP tour:

```bash
python test_model_fixed.py --model_path drl_moa_model.pkl --n_cities 15
```

Parameters:
- `--model_path` - Path to the saved model (default: drl_moa_model.pkl)
- `--n_cities` - Number of cities (default: 20)
- `--input_dim` - Input dimension (default: 4)
- `--hidden_dim` - Hidden dimension (default: 128)

## Deployment

To deploy this application:

### Local Deployment
Run the Streamlit app locally:
```bash
streamlit run app.py
```

### Streamlit Community Cloud Deployment
1. Ensure your repository is pushed to GitHub
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch (main), and specify the main file path (app.py)
6. Click "Deploy"

The repository already includes the necessary configuration files:
- `requirements.txt` - Python dependencies
- `packages.txt` - System-level dependencies
- `.streamlit/` - Streamlit configuration

### Heroku Deployment
To deploy on Heroku:
1. Create a Heroku account
2. Install Heroku CLI
3. Run these commands:
```bash
heroku create your-app-name
git push heroku main
```

### Hyperparameter Tuning

To run hyperparameter tuning:

```bash
# Quick demo with few trials and epochs
python tuning_demo_fixed.py

# Full hyperparameter optimization (takes longer)
python hyperparameter_tuning.py --n_trials 50 --output_model tuned_model.pkl
```

Parameters:
- `--n_trials` - Number of trials for hyperparameter search (default: 50)
- `--study_name` - Name for the Optuna study (default: drl_moa_tuning)
- `--output_model` - Path to save the tuned model (default: drl_moa_tuned.pkl)

### Comparing Models

To compare the original and tuned models:

```bash
python test_tuned_model.py --original_model drl_moa_model.pkl --tuned_model drl_moa_tuned_demo.pkl
```

Parameters:
- `--original_model` - Path to the original model (default: drl_moa_model.pkl)
- `--tuned_model` - Path to the tuned model (default: drl_moa_tuned_demo.pkl)
- `--n_cities` - Number of cities for test instances (default: 20)
- `--n_instances` - Number of test instances (default: 5)

## Hyperparameter Tuning

The hyperparameter tuning process optimizes:

1. **Hidden Dimension**: Size of the hidden layers in the encoder, decoder, and critic networks
2. **Batch Size**: Number of TSP instances used in each training batch
3. **Learning Rate**: Step size for the Adam optimizer

The tuning process:
1. Trains models with different hyperparameter combinations
2. Evaluates each model on validation instances
3. Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to find optimal values
4. Trains a final model with the best hyperparameter combination

Comparison plots between the original and tuned models are saved to the `comparison_plots/` directory.

## Bug Fixes in Fixed Versions

1. **In-place Operations**: The original model had issues with in-place operations during backpropagation. The fixed version avoids these by:
   - Creating new tensor copies before modifying them
   - Properly handling the mask updates in the decoder

2. **Model Loading**: The fixed test script can load models from both .pkl and .pt files:
   - Tries loading with pickle first
   - Falls back to loading the state_dict from .pt files if pickle fails
   - Creates a new model instance with the same architecture

3. **Index Checking**: Added checks for attribute indices in compute_reward to avoid accessing out-of-bounds indices

## Model Architecture

### Encoder
- 1D Convolution Layer that converts city features into embeddings
- Input: City coordinates and attributes
- Output: High-dimensional city embeddings

### Decoder
- GRU-based decoder with attention mechanism
- Constructs a tour one city at a time, ensuring each city is visited exactly once
- Uses masked attention to prevent revisiting cities

### Critic
- Estimates the expected reward for a given state
- Uses the same encoder architecture but with an additional value head

## Example Output

When running the test script, a visualization of the TSP tour will be saved to `tsp_tour.png`.
When comparing models, tour visualizations and performance comparisons are saved to the `comparison_plots/` directory.

## Extensions

This implementation focuses on a single subproblem (weight vector). The full DRL-MOA method from the paper includes:

1. Multiple subproblems with different weight vectors
2. Neighborhood-based parameter transfer between subproblems
3. Various post-processing techniques like local search
4. Support for more objectives

## Citation

If you use this implementation, please cite the original DRL-MOA paper:

```
@article{li2021deep,
  title={Deep Reinforcement Learning for Multi-Objective Optimization},
  author={Li, Kexin and Zhang, Tao and Wang, Rui},
  journal={IEEE Transactions on Cybernetics},
  year={2021}
}
``` 