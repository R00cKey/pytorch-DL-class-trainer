# pytorch-DL-class-trainer

This repository contains the parent class defining utilities for a Deep Learning model training. An example containing an example of inheritance and main is also provided. these are the variables and methods defined inside the mother class:

BaseDLFramework(model, train_dataloader, optimizer, criterion, snapshot_path, model_init_path, best_model_save_path)

## Variables

### model (torch.nn.Module)
Deep Learning model which is going to be trained

### train_dataloader (torch.utils.data.DataLoader)
DataLoader containing the training examples.

### optimizer (torch.optim.Optimizer)
Optimizer used to update the model's parameters 

### criterion (torch.nn.modules.loss._Loss)
Loss function used for training. Note that the objects are implemented (e.g. torch.nn.L1Loss()) not the functionals (e.g. torch.nn.functional.l1_loss)

### snapshot_path (str)
The path to the file where the model's weights and number of epochs that have run are saved and loaded, to store training progress in case of errors during training. The default argument is "snapshot/snapshot.pt"

### model_init_path (str)
The path containing the initialization weights of the model, if present. The default argument is "None"

### best_model_save_path (str)
The file path where the best model's weights, architecture, best loss achieved and optimizer hyperparameters are stores. The default argument is "best_model.pt"

## Internal Methods

### _save_snapshot(epoch)
Backups the training in case of errors, so training can be resumed instead of resetting

### _load_snapshot()
Loads the backup model

### _load_init_model()
Loads the initialized weights into the model

### _save_best_model()
Saves the model which scored the best loss

### _get_optim_hp()
Returns a dictionary containing all the hyperparameters of the optimizer present in the class

### _get_model()
Return the current model state

### _train()
The algorithm of a single training loop. While this is already defined as a pretty common training algorithm, it should be overridden and customized following the requests of the project

## Methods

### run_epoch(max_epochs: int)
Runs the epoch loop max_epochs number of times. In the BaseDLFramework class, this function defines the train -> save_snapshot loop. However, this function should be overridden and customized following the loop logic needed for the project (for example training -> validation -> save_snapshot)

### plot_train_loss_by_epochs(title=None, xlabel=None, ylabel=None, label=None, color=None)
Plots via matplotlib.pyploy.plot the training loss by epochs
