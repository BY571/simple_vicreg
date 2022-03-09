# TODO: 
# add training script with a pretrained // not pretrained VICReg
from doctest import testfile
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

    parser.add_argument("--run_name", type=str, default="VICReg-classification")

    # Checkpoints saving
    parser.add_argument("--checkpoint", type=str, default="models/resnet_1.pth")
    parser.add_argument("--weights", type=str, default="no-freeze", choices=["freeze", "no-freeze"])
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


    transforms = aug.TestTransform()

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)
    
    print("Train Set length: ", len(train_dataset))
    print("Test Set length: ", len(test_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    
    with wandb.init(project="Simple-VICReg", name=args.run_name, config=args):
        backbone, embedding = simple_resnet()
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
        assert missing_keys == [] and unexpected_keys == []

        head = nn.Linear(embedding, 1000)
        head.weight.data.normal_(mean=0.0, std=0.01)
        head.bias.data.zero_()
        model = nn.Sequential(backbone, head)
        model.cuda(gpu)

        if args.weights == "freeze":
            backbone.requires_grad_(False)
            head.requires_grad_(True)

        wandb.watch(model)
        print(model)
        
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)    
        start_epoch = 1

        for epoch in range(start_epoch, args.epochs):
            # Train 
            overall_loss = []
            for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):

                images = images.cuda(gpu, non_blocking=True)
                optimizer.zero_grad()
                prediction = model.forward(images)
                loss = criterion(prediction, target.cuda(gpu, non_blocking=True))
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())

            # Evaluation
            model.eval()
            with torch.no_grad():
                overall_test_loss = []
                for step, (images, target) in enumerate(test_loader, start=epoch * len(test_loader)):

                    images = images.cuda(gpu, non_blocking=True)
                    prediction = model.forward(images)
                    loss = criterion(prediction, target.cuda(gpu, non_blocking=True))
                    overall_test_loss.append(loss.item())
                    # these include the weights!

                print("Epoch: {} -- Train Loss: {:.4f} -- Test Loss: {:.4f}".format(epoch,  np.mean(overall_loss), np.mean(overall_test_loss)))
                wandb.log({"Epoch": epoch,
                           "Train loss": np.mean(overall_loss),
                           "Test loss": np.mean(overall_test_loss)})
                model.train()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
