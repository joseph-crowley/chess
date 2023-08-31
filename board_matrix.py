import chess
import numpy as np
from typing import List, Tuple

piece_map = {
    0: '.',
    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
    -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k'
}

# Convert a 64-element board vector to a sparse representation
def board_vector_to_sparse(board_vector: List[int]) -> List[Tuple[int, int]]:
    return [(pos, piece) for pos, piece in enumerate(board_vector) if piece != 0]

# Convert a sparse representation to a 64-element board vector
def sparse_to_board_vector(sparse_board: List[Tuple[int, int]]) -> List[int]:
    board_vector = [0] * 64
    for pos, piece in sparse_board:
        board_vector[pos] = piece
    return board_vector

# Convert FEN to a 64-character board string
def fen_to_board(fen: str) -> str:
    board = chess.Board(fen)
    board_list = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            board_list.append('.')
        else:
            board_list.append(piece.symbol())
    return ''.join(board_list)

# Convert a 64-character board string to FEN
def board_str_to_fen(board_str, active_color='w', castling='KQkq', en_passant='-', halfmove=0, fullmove=1):
    # Convert the piece placement part
    fen = ''
    empty = 0
    
    for i, char in enumerate(board_str):
        if char == '.':
            empty += 1
        else:
            if empty > 0:
                fen += str(empty)
                empty = 0
            fen += char
        
        if (i + 1) % 8 == 0:
            if empty > 0:
                fen += str(empty)
                empty = 0
            if i < 63:
                fen += '/'
                
    # Add other FEN fields
    fen += f' {active_color} {castling} {en_passant} {halfmove} {fullmove}'
    
    return fen

# Convert a 64-character board string to a 64-element column vector
def board_to_vector(board_str: str) -> List[int]:
    return [piece_map[symbol] for symbol in board_str]

# Convert a 64-element column vector to a 64-character board string
def vector_to_board(board_vector: List[int]) -> str:
    return ''.join([piece_map[int(piece)] for piece in board_vector])

# Convert a 64-element column vector to a FEN string
def vector_to_fen(board_vector: List[int]) -> str:
    board_str = vector_to_board(board_vector)
    board = chess.Board(board_str)
    return board.fen()

def is_valid_board(board_vector):
    valid = np.zeros(board_vector.shape[0], dtype=np.bool_) 
    for i in range(board_vector.shape[0]):
        board_str = vector_to_board(board_vector[i])
        board_fen = board_str_to_fen(board_str)
        board = chess.Board(board_fen)
        valid[i] = board.is_valid()
    return valid


# Apply a list of moves (in UCI format) to a FEN string and return the resulting FEN
def apply_moves_to_fen(fen: str, moves: List[str]) -> str:
    board = chess.Board(fen)
    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
        else:
            raise ValueError(f"Illegal move {move_uci} in position {board.fen()}")
    return board.fen()

# Create a 64 x n matrix and a list of sparse matrices representing board states for each move in the solution
def solution_to_matrix_and_sparse(fen: str, moves: List[str]) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    board_vectors = np.zeros((len(moves), 64), dtype=int)
    sparse_boards = []
    board = chess.Board(fen)
    for i, move_uci in enumerate(moves):
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            board.push(move)
            board_str = fen_to_board(board.fen())
            board_vector = board_to_vector(board_str)
            board_vectors[i, :] = board_vector

            # Create sparse representation
            sparse_board = board_vector_to_sparse(board_vector)
            sparse_boards.append(sparse_board)
        else:
            raise ValueError(f"Illegal move {move_uci} in position {board.fen()}")
    return board_vectors, sparse_boards

    
if __name__ == "__main__":
    # Test the complete program with a single puzzle
    initial_fen = "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17"
    moves = ["e8d7", "a2e6", "d7d8", "f7f8"]
    board_matrix = solution_to_matrix_and_sparse(initial_fen, moves)

    # Output the 64 x n matrix
    print(board_matrix)