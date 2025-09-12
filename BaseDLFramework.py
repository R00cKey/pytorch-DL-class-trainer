import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


class BaseDLFramework:
	def __init__(
		self,
		model: torch.nn.Module,
		train_dataloader: torch.utils.data.DataLoader,
		optimizer: torch.optim.Optimizer,
		criterion: torch.nn.modules.loss._Loss,
		snapshot_path: str = "snapshot/snapshot.pt", #Path and filename
		model_init_path: str = 'None',
		best_model_save_path: str='best_model.pt'
		) -> None:

			self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			self._model = model.to(self._device)
			self.train_data = train_dataloader
			self._optimizer = optimizer
			self._criterion = criterion
			self._epochs_run = 0 #The number of epochs which have already been completed
			self._snapshot_path = snapshot_path
			self._model_init_path = model_init_path
			self._best_model_save_path = best_model_save_path
			self._best_train_loss = np.inf
			self._train_loss_by_epochs = [] #Save training loss in each epoch in order to plot train loss vs epochs

			if os.path.exists(self._snapshot_path):
				print("Loading snapshot")
				self._load_snapshot()

			elif os.path.exists(self._model_init_path):
				print("Loading Initialized model")
				self._load_init_model()

	#Training load and save methods
	def _save_snapshot(self, epoch): ##Backup the training in case of DDP errors, so training can be resumed instead of a reset
		snapshot = {
		  "MODEL_STATE": self._model.state_dict(),
		  "EPOCHS_RUN": epoch,
		  "TRAIN_LOSS_EPOCHS": self._train_loss_by_epochs
		}
		if not os.path.exists(os.path.abspath(os.path.dirname(self._snapshot_path))):
			os.mkdir(os.path.abspath(os.path.dirname(self._snapshot_path)))
		torch.save(snapshot, self._snapshot_path)
		print(f"Epoch {epoch+1} | Training snapshot saved at {self._snapshot_path}")

	def _load_snapshot(self): ##Load the backup at declaration of Trainer class in main()
		snapshot = torch.load(self._snapshot_path, map_location=self._device)
		self._model.load_state_dict(snapshot["MODEL_STATE"])
		self._epochs_run = snapshot["EPOCHS_RUN"]
		self._train_loss_by_epochs = snapshot["TRAIN_LOSS_EPOCHS"]
		print(f"Resuming training from snapshot saved at Epoch {self._epochs_run}")

	#Methods to get and load information on the class
	def _load_init_model(self): ##Initialize the model defined in the class
		init_weights = torch.load(self._model_init_path, map_location=self._device)
		self._model.load_state_dict(init_weights["MODEL_STATE"])
		print(f"Initialized the weights of the model")

	def _save_best_model(self): #Save the model which performed the best
		best_model={
		    "MODEL_STATE": self._model.state_dict(),
		    "MODEL_ARCH": str(self._get_model),
		    "BEST_LOSS": self._best_train_loss,
		    "OPTIM_HYPERPARM": self._get_optim_hp()
		  }
		torch.save(best_model, self._best_model_save_path)

	def _get_optim_hp(self):
		for param_group in self._optimizer.param_groups:
			return {key: value for key, value in param_group.items() if key != "params"}

	def _get_model(self):
		return self._model #Can be used inside print()

	#Methods for training
	def run_epochs(self, max_epochs: int): #max_epochs is the total number of epochs to be run
		for epoch in range(self._epochs_run, max_epochs):
			self._model.train()
			train_loss=self._train()
			if train_loss < self._best_train_loss and self._epochs_run+1 >5:
				self._best_train_loss=train_loss
				print(f"Saving best model at Epoch {self._epochs_run+1}")
				self._save_best_model()
			self._epochs_run+=1
			if epoch % 5 ==0: self._save_snapshot(epoch) ##Backup of training, in case something interrupts the program. Best to run after validation, if present
			

	def _train(self):
		train_loss=0.

		for x, y in self.train_data:
			x, y =x.to(self._device), y.to(self._device)
			self._optimizer.zero_grad()
			outputs = self._model(x)
			loss = self._criterion(outputs, y)
			loss.backward()

			self._optimizer.step()

			train_loss+=loss.item()*x.size(0)

		train_loss = train_loss / len(self.train_data.dataset)
		self._train_loss_by_epochs.append(train_loss)

		return train_loss
		

	#Methods to plot data
	def plot_train_loss_by_epochs(self, title=None, xlabel=None, ylabel=None, label=None, color=None, filename='test_loss.png'):
		fig, ax = plt.subplots()
		ax.plot(np.arange(1, len(self._train_loss_by_epochs)+1), self._train_loss_by_epochs, label=label, color=color)
		if title:
			ax.set_title(title)
		if xlabel:
			ax.set_xlabel(xlabel)
		if ylabel:
			ax.set_ylabel(ylabel)
		if label:
			ax.legend()
		print(f"Saving Train Loss Plot at {os.path.abspath(filename)}")
		fig.savefig(filename)