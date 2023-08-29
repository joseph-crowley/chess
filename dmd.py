import numpy as np
from scipy.linalg import eig
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

class DMDAnalyzer:
    def __init__(self, rank=5):
        self.rank = rank
        self.eigenvalues = None
        self.modes = None
    
    def fit(self, snapshot_matrix):
        """
        Fits the DMD model to the given snapshot matrix
        """
        X = snapshot_matrix[:, :-1]
        X_prime = snapshot_matrix[:, 1:]
        
        # Determine the number of columns (features)
        n_features = X.shape[1]
        
        # Choose the rank based on the number of features
        self.rank = min(n_features, self.rank)

        # Perform SVD and DMD
        if n_features <= 30:  # Threshold for exact SVD can be adjusted
            print("Using exact SVD")
            U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
            X_r = U @ np.diag(Sigma)
            X_r_inv = np.linalg.pinv(X_r)
            A_r = X_r_inv @ X_prime
            V = Vt.T  # Transpose back to get V
        else:
            # Using truncated SVD
            print("Using truncated SVD")
            svd = TruncatedSVD(n_components=self.rank)
            X_r = svd.fit_transform(X)
            X_r_inv = np.linalg.pinv(X_r)
            A_r = X_r_inv @ X_prime @ svd.components_.T  # Corrected line
            V = svd.components_.T  # Transpose to get V
            Sigma = svd.singular_values_

        # Eigen decomposition
        w, v = eig(A_r)

        # Map back to original space
        Phi = X_prime @ V @ np.linalg.pinv(np.diag(Sigma)) @ v  # This line is common to both cases

        self.eigenvalues = w
        self.modes = Phi

        print(f"DMD Eigenvalues: {self.eigenvalues}")
        print(f"DMD Modes shape: {self.modes.shape}")

    def apply(self, initial_condition, time_steps):
        """
        Applies the fitted DMD model to predict future states
        """
        if self.eigenvalues is None or self.modes is None:
            raise ValueError("You must fit the DMD model before applying it.")

        b = np.linalg.pinv(self.modes) @ initial_condition
        future_states = np.zeros((self.modes.shape[0], time_steps), dtype=complex)

        for t in range(time_steps):
            future_states[:, t] = np.dot(self.modes, np.multiply(np.power(self.eigenvalues, t), b))

        return future_states

    def plot_modes(self):
        """
        Visualizes the DMD modes
        """
        if self.modes is None:
            raise ValueError("You must fit the DMD model before plotting the modes.")

        for idx, mode in enumerate(self.modes.T):
            plt.figure()
            plt.plot(np.abs(mode))
            plt.title(f'DMD Mode {idx+1}')
            plt.show()

    def save(self, filename):
        """
        Saves the DMD model to a file
        """
        np.savez(filename, rank=self.rank, eigenvalues=self.eigenvalues, modes=self.modes)

    def load(self, filename):
        """
        Loads the DMD model from a file
        """
        data = np.load(filename)
        self.rank = data['rank']
        self.eigenvalues = data['eigenvalues']
        self.modes = data['modes']

# Example usage
if __name__ == "__main__":
    # Create a random snapshot matrix for demonstration.
    # In real application, replace this with your prepared DMD matrix.
    snapshot_matrix = np.random.rand(100, 20)

    # Initialize the DMD Analyzer
    dmd_analyzer = DMDAnalyzer(rank=5)

    # Fit the DMD model
    dmd_analyzer.fit(snapshot_matrix)

    # Plot the modes
    dmd_analyzer.plot_modes()

    # Predict future states for 10 time steps
    initial_condition = snapshot_matrix[:, 0]
    future_states = dmd_analyzer.apply(initial_condition, 10)