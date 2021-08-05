import argparse

args = argparse.ArgumentParser()
args.add_argument('--epochs', default=10, type=int)
args.add_argument('--network', default='lstm', type=str)
args.add_argument('--dropout', default=0.2, type=float)
args.add_argument('--vocab_size', default=30522, type=int)
args.add_argument('--batch_size', default=32, type=int)

args.add_argument('--bidirectional', default=False, type=bool)

args.add_argument('--reconstruction', default='continuous', type=str)
args.add_argument('--transfer', default='highway', type=str)

args.add_argument('--layer', default=10, type=int)
args.add_argument('--kld_weight', default=0, type=float)
args.add_argument('--input_dim', default=200, type=int)
args.add_argument('--latent_dim', default=200, type=int)
args.add_argument('--hidden_dim', default=200, type=int)

args = args.parse_args()