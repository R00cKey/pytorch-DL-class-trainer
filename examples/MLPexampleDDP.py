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
from BaseDLFrameworkDDP import BaseDLFrameworkDDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import confusion_matrix
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy

def ddp_setup(): #Functions to get the environment variables need to run DDP
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

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
        print(*layers)
        self.fc = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc(input)


class MLPTrainerDDP(BaseDLFrameworkDDP):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        scheduler: torch.optim.lr_scheduler,
        snapshot_path: str = "snapshot/snapshot.pt", #Path and filename
        model_init_path: str = 'None',
        best_model_save_path: str='best_model.pt',
        save_every_epoch: int = 5
        ) -> None:

        #Call the variables from the parent class BaseDLFramework
        super().__init__(model, train_dataloader, optimizer, criterion,
                         snapshot_path, model_init_path, best_model_save_path, save_every_epoch)

        #Adding the new variables
        self._scheduler=scheduler
        self.val_data=val_dataloader
        self._best_val_acc=0
        self._val_acc_by_epochs=[]


    #Modify best model dictionary to store best validation set loss
    def _save_best_model(self): #Save the model which performed the best
        best_model={
            "MODEL_STATE": self._model.state_dict(),
            "MODEL_ARCH": str(self._get_model),
            "BEST_ACC": self._best_val_acc,
            "OPTIM_HYPERPARM": self._get_optim_hp()
          }
        torch.save(best_model, self._best_model_save_path)

    #Add validation loop and include it into the epoch run

    def _validation(self):
        correct_pred=0
        total_pred=0

        for x, y in self.val_data:
            x, y =x.to(self._gpu_id), y.to(self._gpu_id)
            logits = self._model(x)
            predicted_classes = torch.argmax(logits, dim=1)
            correct_pred += (predicted_classes == y).sum().item()
            total_pred += y.size(0)

        val_acc = correct_pred/total_pred
        self._val_acc_by_epochs.append(val_acc)

        return val_acc

    #epochs loop must be updated to include validatoin
    def run_epochs(self, max_epochs: int):
        for epoch in range(self._epochs_run, max_epochs):
            self._model.train()
            b_sz = len(next(iter(self.train_data)))
            print(f"[GPU{self._gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
            train_loss=self._train()
            self._scheduler.step()
            dist.barrier() # Synchronize all ranks before updating variables, running validation on rank 0 and saving

            if self._gpu_id==0:
                if train_loss < self._best_train_loss and self._epochs_run+1 >5:
                    self._best_train_loss=train_loss
                self._model.eval()
                with torch.no_grad():
                    val_acc=self._validation()
                    if val_acc > self._best_val_acc and self._epochs_run+1 >5:
                        self._best_val_acc=val_acc
                        print(f"Saving best model at Epoch {self._epochs_run+1}, with Validation Accuracy: {val_acc}")
                        self._save_best_model()

                self._epochs_run+=1
                if epoch % 5 ==0: self._save_snapshot(epoch)

            dist.barrier()

    #Include test method to get test score
    def eval_accuracy(self, test_data: torch.utils.data.DataLoader):
        self._model.module.load_state_dict(torch.load(self._best_model_save_path)["MODEL_STATE"]) # Load the best-performing model
        correct_pred=0
        total_pred=0
        self._model.eval()
        with torch.no_grad():
            for x, y in test_data:
                x, y =x.to(self._gpu_id), y.to(self._gpu_id)
                logits = self._model(x)
                predicted_classes = torch.argmax(logits, dim=1)
                correct_pred += (predicted_classes == y).sum().item()
                total_pred += y.size(0)

            test_acc = correct_pred/total_pred
            self._model.to(self._gpu_id)
            return test_acc

    #Plot the validation accuracy by epochs
    def plot_val_acc_by_epochs(self, title=None, xlabel=None, ylabel=None, label=None, color=None, filename='Val_acc.png'):
        if self._gpu_id==0:
            fig, ax = plt.subplots()
            ax.plot(np.arange(1, len(self._val_acc_by_epochs)+1), self._val_acc_by_epochs, label=label, color=color)
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            if label:
                ax.legend()
            print(f"Saving Validation Accuracy Plot at {os.path.abspath(filename)}")
            fig.savefig(filename)


    def plot_confusion_matrix(self, test_dataloader, title="Confusion Matrix", class_names=None, colormap=plt.cm.Blues, filename='Confusion_matrix.png'):
        if self._gpu_id==0:
            self._model.module.load_state_dict(torch.load(self._best_model_save_path)["MODEL_STATE"]) # Load the best-performing model
            self._model.eval()

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(self._gpu_id), labels.to(self._gpu_id)
                    outputs = self._model(inputs)
                    _, preds = torch.max(outputs, 1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            cm = confusion_matrix(all_labels, all_preds, labels=np.unique(all_labels))
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots()
            cax = ax.matshow(cm, cmap=colormap)
            fig.colorbar(cax)

            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('True Labels')
            ax.set_title(title)

            # Annotate each cell with the numeric value
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text = f"{cm[i, j]:.2f}"
                    ax.text(j, i, text, ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
            print(f"Saving Confusion Matrix at {os.path.abspath(filename)}")
            fig.savefig(filename)


def main(hidden_dims, lr, epoch_max):
    #Create the dataloaders from the IRIS dataset
    print(f"Running with hidden_dims={hidden_dims}, lr = {lr}, epoch_max = {epoch_max}")
    ddp_setup()
    iris = load_iris()
    x = iris.data        # features (150 samples, 4 features each)
    y = iris.target      # labels (0, 1, 2)
    X, X_test, Y, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size =0.25, random_state=42)
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float)
    X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float)
    X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float)

    batch_size=16
    train_loader= DataLoader(TensorDataset(X_train, torch.tensor(Y_train)), batch_size=batch_size, shuffle=True)
    val_loader= DataLoader(TensorDataset(X_val, torch.tensor(Y_val)), batch_size=batch_size, shuffle=False)
    test_loader= DataLoader(TensorDataset(X_test, torch.tensor(Y_test)), batch_size=batch_size, shuffle=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    model=MLP(4,hidden_dims,3)
    trainer=MLPTrainerDDP(model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.0001),
            snapshot_path=f"snapshot/snapshot_{hidden_dims}_{lr}_{epoch_max}.pt", #Path and filename
            best_model_save_path=f"best_model_{hidden_dims}_{lr}_{epoch_max}.pt")

    trainer.run_epochs(epoch_max)
    trainer.plot_train_loss_by_epochs(title="Training Cross-Entropy by Epochs", xlabel="Epochs", ylabel="Cross-Entropy Loss", filename=f"Train_Loss_{hidden_dims}_{lr}_{epoch_max}.png")
    trainer.plot_val_acc_by_epochs(title="Validation accuracy by Epochs", xlabel="Epochs", ylabel="Accuracy", color="orange", filename=f"Val_Acc_{hidden_dims}_{lr}_{epoch_max}.png")
    print(f"Hidden dims: {hidden_dims}, LR: {lr}, Training Epochs: {epoch_max}, Test Accuracy: {trainer.eval_accuracy(test_loader):.2%}")
    trainer.plot_confusion_matrix(test_loader, class_names=iris.target_names, filename=f"Confusion_matrix_{hidden_dims}_{lr}_{epoch_max}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dims", type=str, default="16")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch_max", type=int, default=50)
    args = parser.parse_args()

    # Convert string to list if needed
    hidden_dims = ast.literal_eval(args.hidden_dims)
    print(hidden_dims)
    main(hidden_dims, args.lr, args.epoch_max)