import numpy as np
import os
import joblib


class ELMPSO:
    """
    Extreme Learning Machine (ELM) model trained with Particle Swarm Optimization (PSO).

    This class loads a pre-trained ELM-PSO model (weights, biases, activation type)
    along with input/output scalers, and provides:
        - Forward prediction of the interaction matrix L
        - Pseudo-inverse computation with optional damping
    """

    def __init__(self, model_dir="~/ros_workspace/src/utils/elmpso_models"):
        """
        Initialize the ELMPSO model by loading parameters and scalers.

        Args:
            model_dir (str): Directory containing the saved model and scalers.
        """
        self.model_dir = os.path.expanduser(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

        # Paths to saved model and scalers
        self.model_path = os.path.join(self.model_dir, "pso_elm_model.npz")
        self.scaler_X_path = os.path.join(self.model_dir, "scaler_X.pkl")
        self.scaler_Y_path = os.path.join(self.model_dir, "scaler_Y.pkl")

        # === Load model parameters ===
        data = np.load(self.model_path, allow_pickle=True)
        self.W_hidden = data['W_hidden']   # Hidden layer weights
        self.b_hidden = data['b']          # Hidden layer biases
        self.W_output = data['W_output']   # Output layer weights
        self.activation_type = str(data['activation'])  # Activation function type

        # === Load input/output scalers ===
        self.scaler_X = joblib.load(self.scaler_X_path)
        self.scaler_Y = joblib.load(self.scaler_Y_path)

    def activation(self, Z):
        """
        Apply the activation function to hidden layer outputs.

        Args:
            Z (np.ndarray): Pre-activation values.

        Returns:
            np.ndarray: Activated values.
        """
        if self.activation_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-Z))
        elif self.activation_type == "tanh":
            return np.tanh(Z)
        elif self.activation_type == "relu":
            return np.maximum(0, Z)
        else:
            raise ValueError("Unknown activation function")

    def predict_L(self, errors):
        """
        Predict the interaction matrix L from feature errors.

        Args:
            errors (np.ndarray): Error vector of shape (2k,), where k is the number of features.

        Returns:
            np.ndarray: Predicted interaction matrix of shape (2k, 6).
        """
        # Normalize input
        x_new = np.array(errors).reshape(1, -1)
        x_scaled = self.scaler_X.transform(x_new)

        # Forward pass through hidden layer
        Omega = self.activation(x_scaled @ self.W_hidden + self.b_hidden)

        # Output in normalized space
        y_scaled = Omega @ self.W_output

        # Inverse transform to original space
        y = self.scaler_Y.inverse_transform(y_scaled)

        # Reshape into interaction matrix (2k x 6)
        num_features = errors.shape[0] // 2
        return y.reshape(2 * num_features, 6)

    def pseudo_inverse(self, L, damping=None):
        """
        Compute the pseudo-inverse of the interaction matrix L.

        Args:
            L (np.ndarray): Interaction matrix of shape (2k, 6).
            damping (float, optional): Damping factor for regularization.
                                       If None, standard pseudo-inverse is used.

        Returns:
            np.ndarray: Pseudo-inverse of L.
        """
        if damping is None:
            return np.linalg.pinv(L)
        else:
            # Damped least-squares pseudo-inverse
            return np.linalg.inv(L.T @ L + damping * np.eye(L.shape[1])) @ L.T