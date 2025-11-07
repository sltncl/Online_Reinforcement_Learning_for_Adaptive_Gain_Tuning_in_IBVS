"""
===============================================================
Standalone PSO-ELM Training Script
===============================================================

This script implements a fully self-contained training pipeline for an 
Extreme Learning Machine (ELM) whose hidden-layer parameters are optimized 
using Particle Swarm Optimization (PSO).

INPUT:
  - trainingDataset.csv : training dataset
  - validationDataset.csv : validation/test dataset

OUTPUT:
  - pso_elm_model.npz : trained model parameters (hidden and output weights)
  - scaler_X.pkl : fitted MinMaxScaler for input normalization
  - scaler_Y.pkl : fitted MinMaxScaler for output normalization

These files will be later used in the thesis project to load and deploy
the trained PSO-ELM network for inference.

===============================================================
"""

# ================================
# Imports
# ================================
import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib

# ================================
# User parameters
# ================================
CSV_PATH = "trainingDataset.csv"      # Path to the training dataset
TEST_PATH = "validationDataset.csv"   # Path to the validation/test dataset
MODEL_PATH = "pso_elm_model.npz"      # File path for saving trained model
SCALER_X_PATH = "scaler_X.pkl"        # File path for saving input scaler
SCALER_Y_PATH = "scaler_Y.pkl"        # File path for saving output scaler

ACTIVATION = "tanh"                   # Activation function {'sigmoid', 'tanh', 'relu'}
m = 50                                # Number of hidden units
PSO_ITERS = 8                         # Number of PSO iterations
SWARM_SIZE = 30                       # Number of particles in the swarm
W_RANGE = (-1.0, 1.0)                 # Range for random weight initialization
INERTIA = 0.8                         # Inertia coefficient for PSO velocity update
C1 = 1.5                              # Cognitive coefficient
C2 = 1.5                              # Social coefficient
SEED = 42                             # Random seed for reproducibility
VERBOSE = True                        # Enable/disable verbose output

# Set random seed
np.random.seed(SEED)


# ================================
# Utility functions
# ================================
def activation(Z, kind='tanh'):
    """Apply the chosen activation function element-wise."""
    if kind == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-Z))
    elif kind == 'tanh':
        return np.tanh(Z)
    elif kind == 'relu':
        return np.maximum(0, Z)
    else:
        raise ValueError(f"Unknown activation function: {kind}")

def encode_Wb(W, b):
    """Flatten weight matrix W and bias vector b into a single concatenated vector."""
    return np.concatenate([W.ravel(), b.ravel()])

def decode_vec(vec, n, m):
    """Decode a flat parameter vector into weight matrix W (n x m) and bias vector b (m,)."""
    w_len = n * m
    W_flat = vec[:w_len]
    b = vec[w_len:w_len + m]
    W = W_flat.reshape((n, m))
    return W, b

def compute_wo(Omega, Y):
    """Compute output layer weights analytically using the Moore–Penrose pseudo-inverse."""
    return np.linalg.pinv(Omega) @ Y

def rmse(Y_true, Y_pred):
    """Compute Root Mean Squared Error (RMSE) between predictions and targets."""
    return np.sqrt(mean_squared_error(Y_true, Y_pred))


# ================================
# Load and preprocess dataset
# ================================
df = pd.read_csv(CSV_PATH)
Y_all = df.values
total_cols = Y_all.shape[1]

# Ensure dataset column count is valid (multiple of 14 as per project specification)
if total_cols % 14 != 0:
    raise ValueError(f"CSV has {total_cols} columns, must be a multiple of 14.")

num_features = total_cols // 14
in_dim = 2 * num_features          # Input dimension
out_dim = 12 * num_features        # Output dimension

# Split dataset into input (X) and output (Y) matrices
X = df.iloc[:, :in_dim].values
Y = df.iloc[:, in_dim:].values

# ================================
# Data normalization
# ================================
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_Y = MinMaxScaler(feature_range=(-1, 1))

# Fit and transform data to the [-1, 1] range
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# Save scalers to disk for future inference
joblib.dump(scaler_X, SCALER_X_PATH)
joblib.dump(scaler_Y, SCALER_Y_PATH)

if VERBOSE:
    print(f"Dataset loaded: N={X.shape[0]}, input_dim={in_dim}, output_dim={out_dim}")
    print("Data normalized to [-1, 1]")


# ================================
# Baseline random ELM
# ================================
def train_elm_random(m, activation_type=ACTIVATION):
    """Train a standard ELM with randomly initialized hidden parameters."""
    # Random initialization of hidden weights and biases
    Wh = np.random.uniform(W_RANGE[0], W_RANGE[1], (in_dim, m))
    bh = np.random.uniform(W_RANGE[0], W_RANGE[1], (m,))
    # Compute hidden layer output
    Omega = activation(X_scaled @ Wh + bh, activation_type)
    # Compute output weights in closed form
    Wo = compute_wo(Omega, Y_scaled)
    # Generate predictions
    Ypred = Omega @ Wo
    # Return model parameters and training RMSE
    return Wh, bh, Wo, rmse(Y_scaled, Ypred)


# ================================
# PSO-based ELM training
# ================================
def pso_elm_train(X, Y, m=m, swarm_size=SWARM_SIZE, iters=PSO_ITERS,
                  activation_type=ACTIVATION, w_range=W_RANGE,
                  inertia=INERTIA, c1=C1, c2=C2, init_around=None):
    """
    Train an Extreme Learning Machine using Particle Swarm Optimization (PSO)
    to optimize hidden layer parameters (weights W and biases b).
    The output layer weights are computed analytically at each iteration.
    """
    n = X.shape[1]
    D = n * m + m  # Dimensionality of the parameter vector (W + b)

    # Initialize swarm either uniformly or around a given seed
    if init_around is None:
        swarm = np.random.uniform(w_range[0], w_range[1], (swarm_size, D))
    else:
        W_seed, b_seed, noise_scale = init_around
        base = encode_Wb(W_seed, b_seed)
        swarm = np.random.normal(loc=base, scale=noise_scale, size=(swarm_size, D))
        swarm = np.clip(swarm, w_range[0], w_range[1])

    # Initialize velocities and best trackers
    vel = np.zeros_like(swarm)
    pbest = swarm.copy()                      # Personal best positions
    pbest_f = np.full((swarm_size,), np.inf)  # Personal best fitness
    gbest = None                              # Global best position
    gbest_f = np.inf                          # Global best fitness

    # Evaluate initial swarm fitness
    for i in range(swarm_size):
        Wh_i, bh_i = decode_vec(swarm[i], n, m)
        Omega_i = activation(X @ Wh_i + bh_i, activation_type)
        Wo_i = compute_wo(Omega_i, Y)
        f_i = rmse(Y, Omega_i @ Wo_i)
        pbest_f[i] = f_i
        if f_i < gbest_f:
            gbest_f = f_i
            gbest = swarm[i].copy()

    if VERBOSE:
        print(f"PSO start - initial best RMSE = {gbest_f:.6f}")

    # ================================
    # PSO iterative optimization loop
    # ================================
    t0 = time.time()
    for it in range(iters):
        for i in range(swarm_size):
            # Random coefficients for stochastic influence
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            # Update velocity and position based on PSO dynamics
            vel[i] = (inertia * vel[i] +
                      c1 * r1 * (pbest[i] - swarm[i]) +
                      c2 * r2 * (gbest - swarm[i]))
            swarm[i] = np.clip(swarm[i] + vel[i], w_range[0], w_range[1])

            # Decode current particle and evaluate its fitness
            Wh_i, bh_i = decode_vec(swarm[i], n, m)
            Omega_i = activation(X @ Wh_i + bh_i, activation_type)
            Wo_i = compute_wo(Omega_i, Y)
            f_i = rmse(Y, Omega_i @ Wo_i)

            # Update personal best
            if f_i < pbest_f[i]:
                pbest_f[i] = f_i
                pbest[i] = swarm[i].copy()

            # Update global best
            if f_i < gbest_f:
                gbest_f = f_i
                gbest = swarm[i].copy()

        if VERBOSE:
            print(f" Iter {it+1}/{iters} - gbest RMSE: {gbest_f:.6f}")

    elapsed = time.time() - t0
    if VERBOSE:
        print(f"PSO finished in {elapsed:.2f}s - best RMSE: {gbest_f:.6f}")

    # Decode best solution and recompute final model
    Wh_best, bh_best = decode_vec(gbest, n, m)
    Omega_best = activation(X @ Wh_best + bh_best, activation_type)
    Wo_best = compute_wo(Omega_best, Y)
    final_rmse = rmse(Y, Omega_best @ Wo_best)

    return Wh_best, bh_best, Wo_best, final_rmse


# ================================
# Baseline ELM training
# ================================
t0 = time.time()
Wh, bh, Wo, baseline_rmse = train_elm_random(m)
elapsed_elm = time.time() - t0
print(f"Baseline random ELM RMSE (train) = {baseline_rmse:.6f}")
print(f"Random ELM training time: {elapsed_elm:.4f} s")

# ================================
# PSO-ELM training
# ================================
t0 = time.time()
Wh_best, bh_best, Wo_best, final_rmse = pso_elm_train(X_scaled, Y_scaled)
elapsed_pso = time.time() - t0
print(f"Final PSO-ELM RMSE (train) = {final_rmse:.6f}")
print(f"PSO-ELM training time: {elapsed_pso:.4f} s")

# ================================
# Save trained model
# ================================
np.savez(MODEL_PATH, W_hidden=Wh_best, b=bh_best, W_output=Wo_best, activation=ACTIVATION)
print("Model and scalers saved successfully.")


# ================================
# Validation on test dataset
# ================================
df_test = pd.read_csv(TEST_PATH)
X_test = df_test.iloc[:, :in_dim].values
Y_test = df_test.iloc[:, in_dim:].values

# Apply the same normalization used during training
X_test_scaled = scaler_X.transform(X_test)
Y_test_scaled = scaler_Y.transform(Y_test)

# Generate predictions on test data
Omega_test = activation(X_test_scaled @ Wh_best + bh_best, ACTIVATION)
Y_pred_test_scaled = Omega_test @ Wo_best

# Compute normalized RMSE on test set
final_rmse_test = rmse(Y_test_scaled, Y_pred_test_scaled)
print(f"RMSE (test set, normalized) = {final_rmse_test:.6f}")

# Inverse transform predictions to original scale
Y_pred_test = scaler_Y.inverse_transform(Y_pred_test_scaled)
final_rmse_test_unscale = rmse(Y_test, Y_pred_test)
print(f"RMSE (test set, unnormalized) = {final_rmse_test_unscale:.6f}")
