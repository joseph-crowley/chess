import pandas as pd
import board_matrix as bm
import h5py
import numpy as np
import json

def load_data(filename, nrows=100):
    """Load data from a CSV file into a Pandas DataFrame.

    Args:
        filename: The name of the CSV file.
        nrows: Number of rows to read from the CSV file.

    Returns:
        A Pandas DataFrame containing the data from the CSV file.
    """
    print(f"Loading data from {filename} with nrows {nrows}")
    df = pd.read_csv(filename, nrows=nrows)
    print(f"Loaded {len(df)} rows with columns {df.columns}")
    print(f'First row:\n{df.iloc[0]}')
    return df

def create_solutions(df, hdf5_filename='solutions.h5'):
    """
    Create a list of solutions from a Pandas DataFrame and save board matrices to an HDF5 file.
    
    Args:
        df: A Pandas DataFrame containing the data.
        hdf5_filename: The name of the HDF5 file to save to.
    """
    with h5py.File(hdf5_filename, 'w') as hf:  # 'w' mode will overwrite existing file
        for index, row in df.iterrows():
            print(f"Processing row {index}\n{row}")
            try:
                initial_fen = row['FEN']
                moves = row['Moves'].split()
            except KeyError:
                print(f"Skipping row {index}: Missing 'FEN' or 'Moves'")
                continue  # Skip this row and continue with the next

            # Determine whose turn it is to move from the FEN string
            turn = 'White' if initial_fen.split(' ')[1] == 'w' else 'Black'

            try:
                board_matrix, sparse_boards = bm.solution_to_matrix_and_sparse(initial_fen, moves)
                
                # Determine the group name based on the turn
                group_name = f"{turn}_to_move"

                # Create the group if it doesn't exist
                if group_name not in hf:
                    hf.create_group(group_name)

                # Save dense matrix to HDF5 file
                hf[f"{group_name}/Puzzle_{row['PuzzleId']}/Dense"] = np.array(board_matrix)
                
                # Serialize sparse matrix to JSON string
                sparse_json_str = json.dumps(sparse_boards)
                
                # Save serialized sparse matrix to HDF5 file
                hf[f"{group_name}/Puzzle_{row['PuzzleId']}/Sparse"] = sparse_json_str
                
            except ValueError as e:
                print(f"An error occurred: {e}")
                print(f"Skipping row {index}")
                continue  # Skip this row and continue with the next

def read_h5(hdf5_filename='solutions.h5', color=None):
    sparse_matrices = []
    with h5py.File(hdf5_filename, 'r') as hf:
        # Determine which groups to iterate over based on the color
        groups_to_check = []
        if color is None:
            groups_to_check = hf.keys()
        elif color.lower() == 'white':
            groups_to_check = [k for k in hf.keys() if 'White' in k]
        elif color.lower() == 'black':
            groups_to_check = [k for k in hf.keys() if 'Black' in k]
        else:
            print(f"Invalid color: {color}. Please use 'white', 'black', or None.")
            return []

        # Loop over each group in the HDF5 file
        for group_name in groups_to_check:
            group = hf[group_name]

            # Loop over each puzzle in the group
            for puzzle_id in group.keys():
                try:
                    sparse_json_str = hf[f"{group_name}/{puzzle_id}/Sparse"][()]
                    sparse_matrix = json.loads(sparse_json_str)
                    sparse_matrices.append(sparse_matrix)
                except KeyError:
                    print(f"Skipping {puzzle_id} in group {group_name}: Missing 'Sparse'")
                    continue  # Skip this puzzle and continue with the next
    return sparse_matrices