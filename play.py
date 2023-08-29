from cVAE import cVAE
from dmd import DMDAnalyzer
from board_matrix import fen_to_board, board_to_vector

import torch
import torch.nn.functional as F
import chess

class KoopmanChessEngine:
    def __init__(self, cVAE_model, dmd_analyzer):
        self.cVAE_model = cVAE_model
        self.dmd_analyzer = dmd_analyzer

    def best_move(self, board):
        best_move = None
        best_score = float('-inf')
    
        for move in board.legal_moves:
            board.push(move)
    
            # Convert the board state to a board vector
            board_str = fen_to_board(board.fen())
            board_vector = board_to_vector(board_str)
    
            # Convert to PyTorch tensor and encode to latent space
            board_tensor = torch.tensor(board_vector, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                z = self.cVAE_model.encoder(board_tensor).numpy()
    
            # Ensure the shape of z matches the DMD modes
            z = z.reshape(-1, self.dmd_analyzer.modes.shape[0])
    
            # Predict the next state in the latent space
            z_next = self.dmd_analyzer.apply(z, 1)  # z is already a numpy array
    
            # Map the predicted state back to the original space
            z_next_tensor = torch.tensor(z_next, dtype=torch.float32)
            with torch.no_grad():
                board_next = self.cVAE_model.decoder(z_next_tensor)
    
            # Evaluate the predicted state
            score = self.evaluate_board_state(board_next.numpy().squeeze(), color=board.turn)
    
            if score > best_score:
                best_score = score
                best_move = move
    
            board.pop()
    
        return best_move

    def evaluate_board_state(self, board_vector, color="white"):
        """ Returns the score of the board state like the eval bar

        Args:
            board_vector (np.array): current board state, 64x1
            color (str, optional): color to evaluate. Defaults to "white".

        Returns:
            _type_: _description_
        """
        # Convert the board to a tensor and encode to latent space
        board_tensor = torch.tensor(board_vector, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            z = self.cVAE_model.encoder(board_tensor)
            z_next = self.dmd_analyzer.apply(z.numpy(), 1)
            board_next = self.cVAE_model.decoder(torch.tensor(z_next, dtype=torch.float32))

        # Material advantage
        material_advantage = 0
        piece_values = {'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -100,
                        'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100}
        for piece in board_vector:
            if piece != 0:
                material_advantage += piece_values[chess.Piece(abs(piece), piece > 0).symbol()]

        # Mobility
        mobility = len(list(board.legal_moves))

        # Combine the factors into a final score
        score = board_next.sum().item() + material_advantage + mobility

        return score

if __name__ == "__main__":
    # Initialize cVAE and DMD models (assuming they are trained)
    cVAE_model = cVAE(hidden_dim=128, latent_dim=32, input_dim=64)
    cVAE_model.load("cVAE_model.pt")

    dmd_analyzer = DMDAnalyzer(rank=32)
    dmd_analyzer.load("dmd_analyzer.npz")

    # Initialize Chess Engine
    engine = KoopmanChessEngine(cVAE_model, dmd_analyzer)

    # Create a chess board
    board = chess.Board()

    # Get the best move according to the Koopman Chess Engine
    best_move = engine.best_move(board)
    print(f"The best move is: {best_move.uci()}")
