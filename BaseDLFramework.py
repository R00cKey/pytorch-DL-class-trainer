import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import logging
from tqdm import tqdm


class BaseDLFramework:
	def __init__(
		self,
		model: torch.nn.Module,
		train_dataloader: torch.utils.data.DataLoader,
		optimizer: torch.optim.Optimizer,
		criterion: torch.nn.modules.loss._Loss,
		snapshot_path: str = 'None', #Path and filename
		model_init_path: str = 'None',
		best_model_save_path: str='None',
		save_every_epoch: int = 5,
		patience: int = 10**9,
		verbosity: int = 0) -> None:

			self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			self._model = model.to(self._device)
			self.train_data = train_dataloader
			self._optimizer = optimizer
			self._criterion = criterion
			self._epochs_run = 0 #The number of epochs which have already been completed
			self._snapshot_path = snapshot_path
			self._model_init_path = model_init_path
			self._best_model_save_path = best_model_save_path
			self._n_save=save_every_epoch
			self._max_patience=patience
			self._patience=0
			self._delta_patience=1.e-4
			self._best_train_loss = np.inf
			self._train_loss_by_epochs = [] #Save training loss in each epoch in order to plot train loss vs epochs
			self._verbosity=verbosity

			self.verbosity_logger = logging.getLogger(__name__)

			if self._verbosity == 0:
				self.verbosity_logger.setLevel(logging.ERROR)
			elif self._verbosity == 1:
				self.verbosity_logger.setLevel(logging.WARNING)
			elif self._verbosity >= 2:
				self.verbosity_logger.setLevel(logging.INFO)

			if self._snapshot_path != 'None':
				if os.path.exists(self._snapshot_path):
					self.verbosity_logger.info("Loading snapshot")
					self._load_snapshot()

			elif os.path.exists(self._model_init_path):
				self.verbosity_logger.info("Loading Initialized model")
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
		self.verbosity_logger.warning(f"Epoch {epoch+1} | Training snapshot saved at {self._snapshot_path}")

	def _load_snapshot(self): ##Load the backup at declaration of Trainer class in main()
		snapshot = torch.load(self._snapshot_path, map_location=self._device)
		self._model.load_state_dict(snapshot["MODEL_STATE"])
		self._epochs_run = snapshot["EPOCHS_RUN"]
		self._train_loss_by_epochs = snapshot["TRAIN_LOSS_EPOCHS"]
		self.verbosity_logger.info(f"Resuming training from snapshot saved at Epoch {self._epochs_run}")

	#Methods to get and load information on the class
	def _load_init_model(self): ##Initialize the model defined in the class
		init_weights = torch.load(self._model_init_path, map_location=self._device)
		self._model.load_state_dict(init_weights["MODEL_STATE"])
		self.verbosity_logger.info(f"Initialized the weights of the model")

	def _save_best_model(self): #Save the model which performed the best
		best_model={
		    "MODEL_STATE": self._model.state_dict(),
		    "MODEL_ARCH": str(self._get_model),
		    "BEST_LOSS": self._best_train_loss,
		    "OPTIM_HYPERPARM": self._get_optim_hp()
		  }
		torch.save(best_model, self._best_model_save_path)
		self.verbosity_logger.warning(f"Saving best model at Epoch {self._epochs_run+1}")

	def _get_optim_hp(self):
		for param_group in self._optimizer.param_groups:
			return {key: value for key, value in param_group.items() if key != "params"}

	def _get_model(self):
		return self._model #Can be used inside print()

	#Methods for training
	def run_epochs(self, max_epochs: int): #max_epochs is the total number of epochs to be run
		epoch_iterator=(tqdm(range(self._epochs_run, max_epochs), desc="Training Progress") if self._verbosity >= 2 
				else range(self._epochs_run, max_epochs))
		for epoch in epoch_iterator:
			self._model.train()
			train_loss=self._train()
	
			if train_loss < self._best_train_loss and self._epochs_run+1 >5:
				self._best_train_loss=train_loss
				if self._best_model_save_path != 'None':
					self._save_best_model()
			if self._verbosity >= 2:
				epoch_iterator.set_postfix(BestTrainLoss=f"{self._best_train_loss:.4f}")
			self._epochs_run+=1
			if abs(self._best_train_loss-train_loss)<self._delta_patience:
				self._patience+=1
			else: self._patience=0

			if self._patience >= self._max_patience:
				self.verbosity_logger.error("Early Stopping Triggered. Best Model has already been saved. Exiting training...")
				break
			if epoch % self._n_save ==0 and self._snapshot_path!='None': self._save_snapshot(epoch) ##Backup of training, in case something interrupts the program. Best to run after validation, if present


			

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
		fig.savefig(filename)