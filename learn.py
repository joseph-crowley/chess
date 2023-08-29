from board_matrix import sparse_to_board_vector
from data import read_h5
from dmd import DMDAnalyzer
from cVAE import cVAE, cVAE_loss

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn.functional as F

import numpy as np
from typing import List, Tuple

def create_snapshot_matrix(sparse_matrices: List[List[Tuple[int, int]]]) -> np.ndarray:
    n = len(sparse_matrices)
    snapshot_matrix = np.zeros((64, n), dtype=int)
    for i, sparse_matrix in enumerate(sparse_matrices):
        board_vector = sparse_to_board_vector(sparse_matrix)
        snapshot_matrix[:, i] = board_vector
    return snapshot_matrix

if __name__ == "__main__":
    # Read sparse matrices and initialize DMD Analyzer
    sparse_matrices = read_h5(color="white")
    dmd_analyzer = DMDAnalyzer(rank=32)

    # Create a list of snapshot matrices
    all_snapshot_matrices = [create_snapshot_matrix(matrix) for matrix in sparse_matrices]
    
    # Concatenate snapshot matrices to form training data
    X_train = np.concatenate(all_snapshot_matrices, axis=1)

    # Create pairs of consecutive states
    X_train_pairs = [(X_train[:, i], X_train[:, i+1]) for i in range(X_train.shape[1]-1)]

    # Convert to PyTorch tensor
    X_train_tensor = torch.tensor([pair[0] for pair in X_train_pairs], dtype=torch.float32)
    X_train_next_tensor = torch.tensor([pair[1] for pair in X_train_pairs], dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, X_train_next_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize and train cVAE
    cVAE_model = cVAE(hidden_dim=128, latent_dim=32, input_dim=64)
    optimizer = Adam(cVAE_model.parameters())

    n_epochs = 50
    for epoch in range(n_epochs):
        for x_t_batch, x_t_plus_1_batch in train_loader:
            optimizer.zero_grad()
            x_t_plus_1_pred, z_mean, z_log_var = cVAE_model(x_t_batch)
            loss = cVAE_loss(x_t_plus_1_batch, x_t_plus_1_pred, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the trained cVAE model
    cVAE_model.save("cVAE_model.pt")

    # Extract latent variables and fit DMD
    with torch.no_grad():
        z = cVAE_model.encoder(X_train_tensor).detach().numpy()

    # Fit the DMD model
    dmd_analyzer.fit(z)

    # Making future state predictions in the latent space
    initial_condition_latent = z[:, 0]  # Take the first latent vector as initial condition
    future_states_latent = dmd_analyzer.apply(initial_condition_latent, 10)  # Predict next 10 states

    # Map future states back to original space using cVAE decoder
    future_states_latent_tensor = torch.tensor(future_states_latent, dtype=torch.float32)

    # Assuming future_states_latent_tensor is your tensor and cVAE_model.latent_dim is the expected size
    latent_dim = cVAE_model.latent_dim
    current_dim = future_states_latent_tensor.shape[1]

    if current_dim < latent_dim:
        # Calculate the number of zeros to pad
        padding = latent_dim - current_dim

        # Pad the tensor along the second dimension (columns)
        future_states_latent_tensor = F.pad(future_states_latent_tensor, (0, padding))

    # Now future_states_latent_tensor should have the correct size and you can pass it to the decoder
    with torch.no_grad():
        future_states_original = cVAE_model.decoder(future_states_latent_tensor)

    # save the dmd_analyzer
    dmd_analyzer.save("dmd_analyzer.npz")

    print("Predicted future states in original space:", future_states_original)
    print(f"Shape: {future_states_original.shape}")

    print(f'Example: {future_states_original[0]}')