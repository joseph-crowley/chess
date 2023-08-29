# Koopman Chess Engine with cVAE

## Overview

This project aims to build a unique chess engine that leverages the power of the Koopman Operator and Conditional Variational Autoencoders (cVAEs) to evaluate chess positions and make decisions. The goal is to approximate the Koopman operator for chess puzzles, which will map any board state closer to a checkmate position.

## Features

- **Data Preprocessing**: Converts chess positions and solutions into a format suitable for machine learning.
- **DMD Analyzer**: Utilizes Dynamic Mode Decomposition (DMD) to approximate the Koopman Operator for an arbitrary snapshot matrix using low-rank approximation.
- **cVAE Model**: Uses a Conditional Variational Autoencoder to encode the high-dimensional chess board into a lower-dimensional latent space.
- **Chess Engine**: Evaluates board positions and decides the best move based on the approximated Koopman operator.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- PyTorch
- h5py
- python-chess
- scikit-learn
- Matplotlib

## Installation

1. Clone this repository.
   ```bash
   git clone https://github.com/joseph-crowley/chess.git
   ```
2. Navigate to the project directory.
   ```bash
   cd chess
   ```
3. Install the required packages.
   ```bash
   pip install numpy pandas pytorch scikit-learn matplotlib h5py python-chess
   ```

## Usage

### Data Preprocessing

1. Place your chess puzzle data in a CSV `puzzles.csv`.
2. Run the preprocessing script to convert this data into HDF5 format.
   ```bash
   python data.py
   ```
   Data will be stored in groups of color/puzzle ID/representation, where representation is either "Dense" or "Sparse"

### Training the cVAE Model

1. Run the cVAE training script.
   ```bash
   python learn.py
   ```
   
### Approximating the Koopman Operator

1. Run the DMD analyzer script to approximate the Koopman Operator for some random matrices.
   ```bash
   python dmd_analyzer.py
   ```

2. Run the cVAE training script to use the puzzles to approximate the Koopman Operator for chess.
    ```bash
    python learn.py
    ```

### Running the Chess Engine

1. Run the main script to start the chess engine.
   ```bash
   python play.py
   ```

## How It Works

### Data Preprocessing

The chess board states and solutions are converted into a numerical representation suitable for machine learning.

### Dynamic Mode Decomposition (DMD)

DMD is used to approximate the Koopman operator which predicts future states of a dynamic system. Here, it's used to predict future chess board states. Using the puzzles as training data, correct future states are closer to winning.

### Conditional Variational Autoencoder (cVAE)

The cVAE model is trained to encode the chess board states into a lower-dimensional latent space and decode them back. This latent space is used for applying the Koopman operator efficiently.

### Chess Engine

The chess engine uses the cVAE and the approximated Koopman operator to evaluate the current board state and predict the board state that is closest to a checkmate position, making that move.

## Contributing

Feel free to fork the project, open a pull request, or submit issues.

## License

This project is licensed under the MIT License. See `LICENSE.md` for details.

---

For more details, please refer to the code and comments. Happy coding!
