import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Custom conditional Variational Autoencoder (cVAE) class
class cVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim=3, encoder=None, decoder=None):
        super(cVAE, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = self.build_encoder()

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = self.build_decoder()

        # Adding the Koopman layer for linear dynamics in latent space
        self.koopman_layer = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
    
    # Encoder network
    def build_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2*self.latent_dim), 
        )
    
    # Decoder network
    def build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
    
    # Sampling from latent space using Gumbel-Softmax
    def sample_from_latent(self, z_logits):
        gumbel_noise = torch.nn.functional.gumbel_softmax(z_logits, tau=1, hard=False)
        return gumbel_noise

    # Full forward pass
    def forward(self, inputs):
        x = self.encoder(inputs)
        z_logits, z_log_var = x.chunk(2, dim=-1)
        z = self.sample_from_latent(z_logits)

        # Apply Koopman layer to enforce linear dynamics
        z_next = self.koopman_layer(z)

        reconstructed = self.decoder(z_next)
        return reconstructed, z_logits, z_log_var

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == '__main__':
    # Hyperparameters
    hidden_dim = 128
    latent_dim = 32

    # Create and compile conditional Variational Autoencoder
    cVAE_model = cVAE(hidden_dim, latent_dim)

    # Optimizer
    optimizer = Adam(cVAE_model.parameters())

    # Summary of the cVAE architecture
    print("Encoder Model")
    print(cVAE_model.encoder)
    print("\nDecoder Model")
    print(cVAE_model.decoder)
    print("\nComplete cVAE Model")
    print(cVAE_model)