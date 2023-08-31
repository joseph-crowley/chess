from board_matrix import sparse_to_board_vector, is_valid_board
from data import read_h5
from dmd import DMDAnalyzer
from cVAE import cVAE

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

def chess_loss(y_true, y_pred, z_logits, z_log_var, invalid_penalty=1000, lambda_reg=0.01):
    reconstruction_loss = F.mse_loss(y_pred, y_true, reduction='none').sum(dim=-1)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_logits**2 - z_log_var.exp(), dim=-1)
    
    # Regularization term
    l2_reg = lambda_reg * torch.sum(z_logits**2)
    
    # Validity check
    invalid_mask = is_valid_board(y_pred)
    invalid_loss = invalid_mask * invalid_penalty

    return (reconstruction_loss + kl_loss + l2_reg + invalid_loss).mean()

def koopman_loss(y_true, y_pred, z_t, z_next_true, z_next_pred, z_mean, z_log_var, koopman_layer, lambda1=1, lambda2=1, lambda3=1, lambda4=1000):    
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(y_pred, y_true, reduction='none').sum(dim=-1)

    # KL-divergence loss
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - z_log_var.exp(), dim=-1)

    # Future prediction loss
    future_loss = F.mse_loss(z_next_pred, z_next_true, reduction='mean')

    # Linearity loss
    linearity_loss = F.mse_loss(z_next_pred, koopman_layer(z_next_true), reduction='mean')
    
    # Validity check
    invalid_board = sum(not s for s in is_valid_board(y_pred))

    return reconstruction_loss.mean() + lambda1 * kl_loss.mean() + lambda2 * future_loss + lambda3 * linearity_loss  + lambda4 * invalid_board

if __name__ == "__main__":
    # Read sparse matrices and initialize DMD Analyzer
    sparse_matrices = read_h5(color="white")

    # Latent dimension = Rank of approximate Koopman operator
    LATENT_DIM = 128


    dmd_analyzer = DMDAnalyzer(rank=LATENT_DIM)

    # Create a list of snapshot matrices
    all_snapshot_matrices = [create_snapshot_matrix(matrix) for matrix in sparse_matrices]

    # create a list of pairs of columns for each snapshot matrix
    for i, snapshot_matrix in enumerate(all_snapshot_matrices):
        all_snapshot_matrices[i] = [(snapshot_matrix[:, j], snapshot_matrix[:, j+1]) for j in range(snapshot_matrix.shape[1]-1)]
    
    # Create a list of pairs of columns for the training data
    X_train_pairs = [] 
    for snapshot_matrix in all_snapshot_matrices:
        X_train_pairs.extend(snapshot_matrix)

    # Convert to PyTorch tensor
    X_train_tensor = torch.tensor([pair[0] for pair in X_train_pairs], dtype=torch.float32)
    X_train_next_tensor = torch.tensor([pair[1] for pair in X_train_pairs], dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, X_train_next_tensor)
    train_loader = DataLoader(train_dataset, batch_size=10**3, shuffle=True)

    # Initialize and train cVAE
    cVAE_model = cVAE(hidden_dim=256, latent_dim=LATENT_DIM, input_dim=64)
    optimizer = Adam(cVAE_model.parameters())

    n_epochs = 500
    for epoch in range(n_epochs):
        for x_t_batch, x_t_plus_1_batch in train_loader:
            optimizer.zero_grad()

            # Encode x_t and x_{t+1} to get z_t and z_{t+1}
            z_t_logits, _ = cVAE_model.encoder(x_t_batch).chunk(2, dim=-1)
            z_t = cVAE_model.sample_from_latent(z_t_logits)

            z_t_plus_1_logits, _ = cVAE_model.encoder(x_t_plus_1_batch).chunk(2, dim=-1)
            z_t_plus_1 = cVAE_model.sample_from_latent(z_t_plus_1_logits)

            # Apply Koopman layer on z_t to get z'
            z_t_prime = cVAE_model.koopman_layer(z_t)

            # Decode z' to get the predicted x_{t+1}
            x_t_plus_1_pred = cVAE_model.decoder(z_t_prime)

            # Compute loss
            loss = koopman_loss(x_t_plus_1_batch, x_t_plus_1_pred, z_t, z_t_plus_1, z_t_prime, z_t_logits, _, cVAE_model.koopman_layer, lambda1=1, lambda2=1, lambda3=1)
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