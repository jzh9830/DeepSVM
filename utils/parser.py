import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lam', type=float, default=0.125,
                        help='trade off.')
    parser.add_argument("--test-ratio",
                        type=float,
                        default=0.1,
                        help="Test data ratio. Default is 0.1.")
    parser.add_argument("--checkpoints", default="./checkpoints/")
    args = parser.parse_args()
    return args