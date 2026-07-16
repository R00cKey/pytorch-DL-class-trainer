import argparse
import ast
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BaseDLFramework import BaseDLFramework
from sklearn.metrics import confusion_matrix
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/MLP_example")

class MLP(nn.Module):
    def __init__(self, input_dim : int , hidden_dims: list, output_dim: int):
        super().__init__()

        layers=[]
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc(input)


def main(hidden_dims, lr, epoch_max, verbosity):
    #Create the dataloaders from the IRIS dataset
    print(f"Running with hidden_dims={hidden_dims}, lr = {lr}, epoch_max = {epoch_max}")
    iris = load_iris()
    x = iris.data        # features (150 samples, 4 features each)
    y = iris.target      # labels (0, 1, 2)
    X, X_test, Y, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size =0.25, random_state=42)
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float)
    X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float)
    X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float)

    train_loader= DataLoader(TensorDataset(X_train, torch.tensor(Y_train)), batch_size=16, shuffle=True)
    val_loader= DataLoader(TensorDataset(X_val, torch.tensor(Y_val)), batch_size=16, shuffle=False)
    test_loader= DataLoader(TensorDataset(X_test, torch.tensor(Y_test)), batch_size=16, shuffle=False)

    writer = SummaryWriter()
    model=MLP(4,hidden_dims,3)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    trainer=BaseDLFramework(model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            train_criterion=nn.CrossEntropyLoss(),
            val_criterion=nn.CrossEntropyLoss(),
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0001),
            verbosity=verbosity,
            snapshot_path=f"snapshot/snapshot_{hidden_dims}_{lr}_{epoch_max}.pt", #Path and filename
            best_model_save_path=f"best_model_{hidden_dims}_{lr}_{epoch_max}.pt",
            writer=writer
    )
    if verbosity == 1: print(trainer._get_model())
    trainer.run_epochs(epoch_max)
    #trainer.plot_train_loss_by_epochs(title="Training Cross-Entropy by Epochs", xlabel="Epochs", ylabel="Cross-Entropy Loss", filename=f"Train_Loss_{hidden_dims}_{lr}_{epoch_max}.png")
    #trainer.plot_val_loss_by_epochs(title="Training Cross-Entropy by Epochs", xlabel="Epochs", ylabel="Accuracy", color="orange", filename=f"Val_Loss_{hidden_dims}_{lr}_{epoch_max}.png")
    print(f"Hidden dims: {hidden_dims}, LR: {lr}, Training Epochs: {epoch_max}, Test Cross-Entropy: {trainer.test(test_loader):.4f}")

    writer.close() #Close Tensorboard SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hc',
                    nargs='+',
                    type=int,
                    help ='List of the hidden features in each Layer')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch_max", type=int, default=50)
    parser.add_argument("--verbosity", type=int, default=0)
    args = parser.parse_args()

    main(args.hc, args.lr, args.epoch_max, args.verbosity)
