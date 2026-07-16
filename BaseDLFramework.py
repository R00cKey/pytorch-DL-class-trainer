import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



class BaseDLFramework:
	def __init__(
		self,
		model: torch.nn.Module,
		train_dataloader: torch.utils.data.DataLoader,
		optimizer: torch.optim.Optimizer,
		train_criterion: torch.nn.modules.loss._Loss,
		scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,

		save_every_n_epochs: int = 5,
		patience: int = 10**9,

		val_dataloader: torch.utils.data.DataLoader | None = None,
		val_criterion: torch.nn.modules.loss._Loss | None = None,

		verbosity: int = 0,
		writer: torch.utils.tensorboard.SummaryWriter | None = None,

		snapshot_path: str | None = None,
		model_init_path: str | None = None,
		best_model_save_path: str | None = None) -> None:

			"""Initialize the trainer.

			Args:
				model: Neural network model to train.
				train_dataloader: DataLoader containing the training dataset.
				
				optimizer: Optimizer used to update model parameters.
				train_criterion: Loss function used during training.
				scheduler: Learning rate scheduler. Set to None if not used.

				save_every_n_epochs: Save a checkpoint every N epochs.
				patience: Number of epochs to wait for validation improvement before early stopping.

				val_dataloader: DataLoader containing the validation dataset. Set to None to disable validation.
				val_criterion: Loss function used during validation. If None, 'train_criterion' will be reused.

				verbosity: Verbosity level (0 = silent, 1 = save operations, 2 = epochs progress bar).
				writer: Optional Tensorboard writer for visualization

				snapshot_path: Path and filename where training snapshots/checkpoints are saved.
				model_init_path: Path to a pretrained model checkpoint to load before training.
				best_model_save_path: Path where the best-performing model is saved.
			"""

			self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			self._model = model.to(self._device)
			self.train_data = train_dataloader
			self.val_data = val_dataloader
			self._optimizer = optimizer
			self._scheduler=scheduler
			self._train_criterion = train_criterion
			self._val_criterion = val_criterion
			self._epochs_run = 0 #The number of epochs which have already been completed
			self._snapshot_path = snapshot_path
			self._model_init_path = model_init_path
			self._best_model_save_path = best_model_save_path
			self._n_save=save_every_n_epochs
			self._max_patience=patience
			self._patience=0
			self._delta_patience=1.e-4
			self._best_train_loss = np.inf
			self._best_val_loss = np.inf
			self._train_loss_by_epochs = [] #Save training loss in each epoch in order to plot train loss vs epochs
			self._val_loss_by_epochs = [] #Save validation loss in each epoch in order to plot train loss vs epochs
			self._verbosity=verbosity
			self._writer = writer

			self.verbosity_logger = logging.getLogger(__name__)

			if self._verbosity == 0:
				self.verbosity_logger.setLevel(logging.ERROR)
			elif self._verbosity == 1:
				self.verbosity_logger.setLevel(logging.WARNING)
			elif self._verbosity >= 2:
				self.verbosity_logger.setLevel(logging.INFO)

			if self._snapshot_path:
				if os.path.exists(self._snapshot_path):
					self.verbosity_logger.info("Loading snapshot")
					self._load_snapshot()

			elif self._model_init_path:
				if os.path.exists(self._model_init_path):
					self.verbosity_logger.info("Loading Initialized model")
					self._load_init_model()

	#Training load and save methods
	def _save_snapshot(self, epoch): ##Backup the training in case of DDP errors, so training can be resumed instead of a reset
		snapshot = {
		  "MODEL_STATE": self._model.state_dict(),
		  "EPOCHS_RUN": epoch,
		  "TRAIN_LOSS_EPOCHS": self._train_loss_by_epochs
		}
		if self.val_data:
			snapshot.update({"VAL_LOSS_EPOCHS": self._val_loss_by_epochs})
		if not os.path.exists(os.path.abspath(os.path.dirname(self._snapshot_path))):
			os.mkdir(os.path.abspath(os.path.dirname(self._snapshot_path)))
		torch.save(snapshot, self._snapshot_path)
		self.verbosity_logger.warning(f"Epoch {epoch+1} | Training snapshot saved at {self._snapshot_path}")

	def _load_snapshot(self): ##Load the backup at declaration of Trainer class in main()
		snapshot = torch.load(self._snapshot_path, map_location=self._device)
		self._model.load_state_dict(snapshot["MODEL_STATE"])
		self._epochs_run = snapshot["EPOCHS_RUN"]
		self._train_loss_by_epochs = snapshot["TRAIN_LOSS_EPOCHS"]
		if 'VAL_LOSS_EPOCHS' in snapshot:
			self._val_loss_by_epochs = snapshot['VAL_LOSS_EPOCHS']
		self.verbosity_logger.info(f"Resuming training from snapshot saved at Epoch {self._epochs_run}")

	#Methods to get and load information on the class
	def _load_init_model(self): ##Initialize the model defined in the class
		init_weights = torch.load(self._model_init_path, map_location=self._device)
		self._model.load_state_dict(init_weights["MODEL_STATE"])
		self.verbosity_logger.info("Initialized the weights of the model")

	def _save_best_model(self): #Save the model which performed the best
		best_model={
		    "MODEL_STATE": self._model.state_dict(),
		    "MODEL_ARCH": str(self._get_model),
		    "BEST_TRAIN_LOSS": self._best_train_loss,
		    "OPTIM_HYPERPARM": self._get_optim_hp()
		  }
		if self.val_data:
			best_model.update({"BEST_VAL_LOSS": self._best_val_loss})
		torch.save(best_model, self._best_model_save_path)
		self.verbosity_logger.warning(f"Saving best model at Epoch {self._epochs_run+1}")

	def _get_optim_hp(self):
		for param_group in self._optimizer.param_groups:
			return {key: value for key, value in param_group.items() if key != "params"}

	def _get_model(self):
		return self._model

	#Methods for training
	def run_epochs(self, max_epochs: int): #max_epochs is the total number of epochs to be run
		epoch_iterator=(tqdm(range(self._epochs_run, max_epochs), desc="Training Progress") if self._verbosity >= 2 
				else range(self._epochs_run, max_epochs))
		for epoch in epoch_iterator:
			#Training
			self._model.train()
			train_loss=self._train()
			self._train_loss_by_epochs.append(train_loss)

			if self._writer: self._writer.add_scalar("Loss/train", train_loss, epoch)

			if self._scheduler: self._scheduler.step()

			if train_loss < self._best_train_loss and self._epochs_run+1 >5:
				self._best_train_loss=train_loss
				if self._best_model_save_path and self.val_data is None:
					self._save_best_model()

			#Optional Validation (if validation dataloader provided)
			if self.val_data:
				self._model.eval()
				with torch.no_grad():
					val_loss=self._validation()

					self._val_loss_by_epochs.append(val_loss)
					if self._writer: self._writer.add_scalar("Loss/val", val_loss, epoch)

					if val_loss < self._best_val_loss and self._epochs_run+1 >5:
						self._best_val_loss=val_loss
						if self._best_model_save_path:
							self._save_best_model()

			if self._verbosity >= 2:
				postfix = {"BestTrainLoss": f"{self._best_train_loss:.4f}"}
				if self.val_data:
					postfix["BestValLoss"] = f"{self._best_val_loss:.4f}"
				epoch_iterator.set_postfix(**postfix)
			self._epochs_run+=1
			if self.val_data is None:
				if abs(self._best_train_loss-train_loss)<self._delta_patience:
					self._patience+=1
				else: self._patience=0
			else:
				if abs(self._best_val_loss-val_loss)<self._delta_patience:
					self._patience+=1
				else: self._patience=0

			if self._writer: self._writer.flush()
			if self._patience >= self._max_patience:
				self.verbosity_logger.error("Early Stopping Triggered. Best Model has been saved. Exiting training...")
				break
			if epoch % self._n_save ==0 and self._snapshot_path: self._save_snapshot(epoch) ##Backup in case something interrupts the program.

	def _train(self):
		train_loss=0.

		for x, y in self.train_data:
			x, y =x.to(self._device), y.to(self._device)
			self._optimizer.zero_grad()
			outputs = self._model(x)
			loss = self._train_criterion(outputs, y)

			loss.backward()

			self._optimizer.step()

			train_loss+=loss.item()*x.size(0)

		train_loss = train_loss / len(self.train_data.dataset)
		return train_loss

	def _validation(self):
		val_loss=0.
		for x, y in self.val_data:
			x, y =x.to(self._device), y.to(self._device)
			outputs = self._model(x)
			if self._val_criterion: loss = self._val_criterion(outputs, y) #val_criterion can be None
			else: loss = self._train_criterion(outputs, y)
			val_loss+=loss.item()*x.size(0)

		val_loss = val_loss / len(self.val_data.dataset)

		return val_loss
	
	#Test method to get test score
	def test(self, test_data: torch.utils.data.DataLoader):
		if self._best_model_save_path: # Load the best-performing model if present
			self._model.load_state_dict(torch.load(self._best_model_save_path)["MODEL_STATE"])
		test_loss=0.
		self._model.to('cpu')
		self._model.eval()
		for x, y in test_data:
			x, y =x.to(self._device), y.to(self._device)
			outputs = self._model(x)
			if self._val_criterion: loss = self._val_criterion(outputs, y) #val_criterion can be None
			else: loss = self._train_criterion(outputs, y)
			test_loss+=loss.item()*x.size(0)

		test_loss = test_loss / len(test_data.dataset)
		if self._writer: self._writer.add_scalar("Loss/test", test_loss)
		self._model.to(self._device)
		if self._writer: self._writer.flush()
		return test_loss

	#Methods to plot data
	def plot_train_loss_by_epochs(self, title=None, xlabel=None, ylabel=None, label=None, color=None, filename='train_loss_by_epochs.png') -> None:
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
	
	def plot_val_loss_by_epochs(self, title=None, xlabel=None, ylabel=None, label=None, color=None, filename='val_loss_by_epochs.png') -> None:
		fig, ax = plt.subplots()
		ax.plot(np.arange(1, len(self._val_loss_by_epochs)+1), self._val_loss_by_epochs, label=label, color=color)
		if title:
			ax.set_title(title)
		if xlabel:
			ax.set_xlabel(xlabel)
		if ylabel:
			ax.set_ylabel(ylabel)
		if label:
			ax.legend()
		fig.savefig(filename)