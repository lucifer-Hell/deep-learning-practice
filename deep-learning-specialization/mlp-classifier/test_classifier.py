import numpy as np
from sklearn.model_selection import train_test_split
import torch
from ClassifierNN import ClassifierNN
# Utitly function to generate circle data and generate dataset on it
def generate_circle_band_dataset(n_points=500, inner_radius=0.7, outer_radius=1.2, seed=42):
    """
    Generate dataset of points (x1, x2) labeled 1 if they lie within a circular ring (inner_radius < r < outer_radius),
    else labeled 0.
    
    Returns:
        X: shape (2, n_points) — feature matrix
        y: shape (1, n_points) — labels
    """
    rng = np.random.default_rng(seed)
    
    X = rng.uniform(low=-1.5*outer_radius, high=1.5*outer_radius, size=(2, n_points))  # shape (2, n_points)
    r = np.sqrt(X[0]**2 + X[1]**2)
    y = ((r > inner_radius) & (r < outer_radius)).astype(int)
    
    return X.T, y.reshape(-1,1)

X, Y = generate_circle_band_dataset(n_points=10000, inner_radius=5, outer_radius=10, seed=42)

# Convert to float32 (PyTorch expects float32 not float64)
X = X.astype(np.float32)
Y = Y.astype(np.float32)
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# Create Model and train 
model = ClassifierNN(
    hidden_layers=[64,32,16,8,4,2],
    input_parameters_len=X.shape[1],
    output_size=1,
    learning_rate=1e-3
)
# train 
model.train_model(X_train_tensor,y_train_tensor,epochs=30000)

model.test(X_test_tensor[1],y_test_tensor[1])
