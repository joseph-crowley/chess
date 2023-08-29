from board_matrix import sparse_to_board_vector
from data import read_h5
from dmd import DMDAnalyzer 

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
    sparse_matrices = read_h5()

    # Initialize the DMD Analyzer
    dmd_analyzer = DMDAnalyzer(rank=50)
    
    for matrix in sparse_matrices[:5]:
        # Create the snapshot matrix
        snapshot_matrix = create_snapshot_matrix(matrix)
        
        # Fit the DMD model to the snapshot matrix
        dmd_analyzer.fit(snapshot_matrix)
        
        # Visualize the DMD modes
        dmd_analyzer.plot_modes()
        
        # Predict future states for 10 time steps
        initial_condition = snapshot_matrix[:, 0]
        future_states = dmd_analyzer.apply(initial_condition, 10)