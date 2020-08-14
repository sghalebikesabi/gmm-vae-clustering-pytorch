from __future__ import print_function
import argparse
import sys
import torch
import torch.utils.data
from torch import optim
from torchvision import datasets, transforms

from gmmvae import VAE
from train import train
from utils import init_weights, open_file


def parse_args(argv):
    """Parses the arguments for the model.
    """
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--z-dim', type=int, default=64, help='dimension of latent factor z (default: 5)')
    parser.add_argument('--r-cat-dim', type=int, default=10, help='dimension of latent factor s (default: 5)')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    return(args)


def main(args): 

    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    # binarize data

    model_params_dict = {
        'cuda': args.cuda,
        'input_dim': 784, 
        'r_cat_dim': args.r_cat_dim,
        'z_dim': args.z_dim, 
        'h_dim': 512, 
    }

    model = VAE(model_params_dict).to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    log_file = 'logs/gmvae.log'
    f = open_file(log_file)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, test_loader, optimizer, device, f)

    if f is not None: 
        f.close()        
    


if __name__ == "__main__":
    args = parse_args(sys.argv[1:]) # sys.argv[0] is file name
    main(args)
