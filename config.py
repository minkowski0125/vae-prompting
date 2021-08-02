import argparse

args = argparse.ArgumentParser()
args.add_argument('--epochs', default=10, type=int)
args.add_argument('--network', default='lstm', type=str)
args.add_argument('--dropout', default=0.2, type=float)
args.add_argument('--vocab_size', default=10000, type=int)
args.add_argument('--batch_size', default=128, type=int)

args.add_argument('--bidirectional', default=False, type=bool)

args.add_argument('--layer', default=6, type=int)
args.add_argument('--kld_weight', default=0.5, type=float)
args.add_argument('--input_dim', default=200, type=int)
args.add_argument('--latent_dim', default=400, type=int)
args.add_argument('--hidden_dim', default=400, type=int)

args = args.parse_args()