from pathlib import Path
import argparse
import os
import sys
import numpy as np

import wandb
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.datasets as datasets

import augmentations as aug

from resnet import simple_resnet, resnet50, resnet34


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    parser.add_argument("--run_name", type=str, default="VICReg-Image")

    # Checkpoints saving
    parser.add_argument("--save-every", type=int, default=5, help="")

    # Model
    # parser.add_argument("--arch", type=str, default="resnet50",
    #                     help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="256-256-256", # for ImageNetdefault="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=50,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=512,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Default learning rate when using a standard optimizer")
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    args = parser.parse_args()
    return args


def main(args):
    torch.backends.cudnn.benchmark = True
    print(args)
    gpu = torch.device(args.device)

    if not os.path.exists("models/"):
        os.makedirs('models/')


    transforms = aug.TrainTransform()

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    print("Train Set length: ", len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    
    with wandb.init(project="Simple-VICReg", name=args.run_name, config=args):
        model = VICReg(args).cuda(gpu)
        wandb.watch(model)
        print(model)
        
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)    
        start_epoch = 1

        for epoch in range(start_epoch, args.epochs):
            overall_loss, sim_losses, std_losses, cov_losses = [], [], [], []
            for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)

                optimizer.zero_grad()
                loss, (sim_loss, std_loss, cov_loss) = model.forward(x, y)
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
                # these include the weights!
                sim_losses.append(sim_loss.item())
                std_losses.append(std_loss.item())
                cov_losses.append(cov_loss.item())
            print("Epoch: {} -- Loss: {:.4f}".format(epoch, np.mean(overall_loss)))
            wandb.log({"Epoch": epoch,
                       "Train loss": np.mean(overall_loss),
                       "Similarity loss": np.mean(sim_losses),
                       "Std loss": np.mean(std_losses),
                       "Covariance loss": np.mean(cov_losses)})
            if epoch % args.save_every == 0:
                torch.save(model.module.backbone.state_dict(), "models/resnet50_{}.pth".format(epoch))


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        # only loading one resnet encoder! Remember this could be two different networks, with different architecture!
        self.backbone, self.embedding = simple_resnet()

        # expander layers
        # could be also two expanders
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss, (self.args.sim_coeff * repr_loss, self.args.std_coeff * std_loss, self.args.cov_coeff * cov_loss)


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
