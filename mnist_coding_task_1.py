""" MNIST Classification """

import argparse
import random
import numpy as np
import torch
from utils import run

MANUAL_SEED = 42

def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Disable CUDNN features for full determinism
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True






def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description='MNIST Classification')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU id to use (use -1 for CPU)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='Testing batch size')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Learning rate')
    parser.add_argument('--eval-every', type=int, default=5,
                        help='Frequency of evaluation (in epochs)')
    parser.add_argument('--dropout1', type=float, default=0.15,
                        help='Dropout rate for first dropout layer')
    parser.add_argument('--dropout2', type=float, default=0.15,
                        help='Dropout rate for second dropout layer')
    parser.add_argument('--hidden_1', type=int, default=32,
                        help='Units in the first hidden layer')
    parser.add_argument('--hidden_2', type=int, default=64,
                        help='Units in the second hidden layer')
    parser.add_argument('--hidden_3', type=int, default=128,
                        help='Units in the third hidden layer')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Flag to save the trained model')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    parser.add_argument('--plot_loss', action='store_true', default=False,
                        help='Plot loss curves during training')
    return parser.parse_args()


def main():
    """Main function to run the MNIST classification process."""
    # Defining settings
    args_dict = {
        'gpu': 0,
        'batch_size': 64,
        'test_batch_size': 1000,
        'epochs': 50,
        'lr': 0.43,
        'val_split': 0.2,
        'dropout1': 0,
        'dropout2': 0.17,
        'hidden_1': 48,
        'hidden_2': 48,
        'hidden_3': 112,
        'patience': 15,
        'save_model': False,
        'plot_loss': False
    }

    args = argparse.Namespace(**args_dict)

    set_seed(MANUAL_SEED)

    # Choose device based on GPU availability
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
    print(f'Using device: {device}')

    # Run the training and evaluation process
    train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy = run(args, device)

    # Print final results
    print(
        f"Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, "
        f"Test loss: {test_loss:.4f}\n"
        f"Train accuracy: {train_accuracy:.2f}, Validation accuracy: {val_accuracy:.2f}, "
        f"Test accuracy: {test_accuracy:.2f}"
    )

if __name__ == '__main__':
    main()
