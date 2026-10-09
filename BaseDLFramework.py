import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import os
import logging
from tqdm import tqdm
import torch.utils.tensorboard
from .exceptions import ModelCollapseError


class BaseDLFramework:
	def __init__(
		self,
		device: torch.device | str,
		model: torch.nn.Module,
		train_dataloader: torch.utils.data.DataLoader,
		optimizer: torch.optim.Optimizer,
		train_criterion: torch.nn.modules.loss._Loss,
		scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,

		save_every_n_epochs: int = 5,
		patience: int = 10**9,
		delta_patience: float = 1.e-5,

		val_dataloader: torch.utils.data.DataLoader | None = None,
		val_criterion: torch.nn.modules.loss._Loss | None = None,

		verbosity: bool = False,
		writer: torch.utils.tensorboard.SummaryWriter | None = None,

		snapshot_filename: str | None = None,
		best_model_filename: str | None = None) -> None:

			"""Initializes the trainer class for Deep Learning.

			Args:
				model: Neural network model to train.
				train_dataloader: DataLoader containing the training dataset.
				
				optimizer: Optimizer used to update model parameters.
				train_criterion: Loss function used during training.
				scheduler: Learning rate scheduler. Set to None if not used.

				save_every_n_epochs: Save a model snapshot every N epochs.
				patience: Number of epochs to wait for validation improvement before early stopping.
				delta_patience: Improvement threshold

				val_dataloader: DataLoader containing the validation dataset. Set to None to disable validation.
				val_criterion: Loss function used during validation. If None, 'train_criterion' will be reused.

				verbosity: Verbosity level (0 = silent, 1 = progress bar).
				writer: Optional Tensorboard writer for visualization

				snapshot_filename: Training snapshots will be saved at snapshots/snapshot_filename.
				best_model_save_name: Best-performing model will be saved at best_models/best_model_save_name.
			"""

			self._device = torch.device(device)
			self._model = model.to(self._device)
			self.train_data = train_dataloader
			self.val_data = val_dataloader
			self._optimizer = optimizer
			self._scheduler=scheduler
			self._train_criterion = train_criterion
			self._val_criterion = val_criterion
			self._epochs_run = 0 #The number of epochs which have already been completed
			self._snapshot_path = os.path.join("snapshots/", snapshot_filename) if snapshot_filename else None
			self._best_model_save_path = os.path.join("best_models/", best_model_filename) if best_model_filename else None
			self._n_save=save_every_n_epochs
			self._max_patience=patience
			self._patience=0
			self._delta_patience=delta_patience
			self._best_train_loss = np.inf
			self._best_patience_loss = np.inf
			self._best_val_loss = np.inf
			self._train_loss_by_epochs = [] #Save training loss in each epoch in order to plot train loss vs epochs
			self._val_loss_by_epochs = [] #Save validation loss in each epoch in order to plot train loss vs epochs
			self._verbosity=verbosity
			self._writer = writer

			self.verbosity_logger = logging.getLogger(__name__)
			self.verbosity_logger.setLevel(logging.ERROR)
			if verbosity:
				#self.verbosity_logger.setLevel(logging.WARNING)
				self.verbosity_logger.setLevel(logging.INFO)
				
			if self._snapshot_path:
				if os.path.exists(self._snapshot_path):
					self.verbosity_logger.info("Loading snapshot")
					self._load_snapshot()

	def training_params(self):
		"""
		Prints parameters related to the Training and Validation processes.
		The parameters are:
			-Training Criterion.
			-Optimizer.
			-Scheduler.
			-Patience.
			-Validation Criterion.
			-Snapshot Save Path.
			-Bast Model Save Path.
		"""
		print("-------------------")
		print("TRAINING PARAMETERS")
		print("-------------------")
		print(f"Criterion:\t{self._train_criterion}")
		print(f"Optimizer:\t{self._optimizer}")
		print(f"Scheduler:\t{self._scheduler}")
		print(f"Patience:\t{self._max_patience}, Delta:\t{self._delta_patience}")
		print("\n")
		if self.val_data is None:
			print("NO VALIDATION")
		else:
			print("---------------------")
			print("VALIDATION PARAMETERS")
			print("---------------------")
			if self._val_criterion is None:
				print(f"Criterion:\t{self._train_criterion}")
			else:
				print(f"Criterion:\t{self._val_criterion}")
		print("------------------")
		print("SAVING DIRECTORIES")
		print("------------------")
		print(f"Snapshot Path:\t{self._snapshot_path}")
		print(f"Best Model Save Path:\t{self._best_model_save_path}\n")

	#Training load and save methods
	def _save_snapshot(self, epoch: int):
		"""
		Backup the training in case of errors, so training can be resumed instead of a full reset
		Args:
			epoch: The epoch at which the snapshot is made.
		"""
		snapshot = {
		  "MODEL_STATE": self._model.state_dict(),
		  "EPOCHS_RUN": epoch,
		  "TRAIN_LOSS_EPOCHS": self._train_loss_by_epochs,
		  "PATIENCE": self._patience,
		}
		if self.val_data:
			snapshot.update({"VAL_LOSS_EPOCHS": self._val_loss_by_epochs})
		if not os.path.exists(os.path.abspath(os.path.dirname(self._snapshot_path))):
			os.makedirs(os.path.abspath(os.path.dirname(self._snapshot_path)))
		torch.save(snapshot, self._snapshot_path)

	def _load_snapshot(self):
		"""
		Load the backup. Used at declaration of the trainer class.
		"""
		snapshot = torch.load(self._snapshot_path, map_location=self._device)
		self._model.load_state_dict(snapshot["MODEL_STATE"])
		self._epochs_run = snapshot["EPOCHS_RUN"]
		self._train_loss_by_epochs = snapshot["TRAIN_LOSS_EPOCHS"]
		self._patience = snapshot["PATIENCE"]
		if 'VAL_LOSS_EPOCHS' in snapshot:
			self._val_loss_by_epochs = snapshot['VAL_LOSS_EPOCHS']
		self.verbosity_logger.warning(f"Resuming training from snapshot saved at Epoch {self._epochs_run}")

	#Methods to get and load information on the class
	def load_model_weights(self, model_path: str):
		"""
		Loads the model weights into the model.
		"""
		init_weights = torch.load(model_path, map_location=self._device)
		self._model.load_state_dict(init_weights["MODEL_STATE"])
		self.verbosity_logger.warning(f"Initialized weights of the model at {model_path}")

	def _save_best_model(self): #Save the model which performed the best
		"""
		Saves a dictionary containing:
		 -Model state
		 -Model Architecture
		 -Best Training Loss
		 -Best Validation Loss
		 -Optimizer Hyperparameters
		of the best-trained model at "self._best_model_save_path"
		"""

		best_model={
		    "MODEL_STATE": self._model.state_dict(),
		    "MODEL_ARCH": str(self._get_model),
		    "BEST_TRAIN_LOSS": self._best_train_loss,
		    "OPTIM_HYPERPARM": self._get_optim_hp()
		  }
		if self.val_data:
			best_model.update({"BEST_VAL_LOSS": self._best_val_loss})
		if not os.path.exists(os.path.abspath(os.path.dirname(self._best_model_save_path))):
			os.makedirs(os.path.abspath(os.path.dirname(self._best_model_save_path)))
		torch.save(best_model, self._best_model_save_path)

	def _get_optim_hp(self):
		"""
		Gets self._optimizer hyperparameters
		"""
		for param_group in self._optimizer.param_groups:
			return {key: value for key, value in param_group.items() if key != "params"}

	def _get_model(self):
		"""
		Gets the model used in the training class
		"""
		return self._model

	#Methods for training
	def run_epochs(self, max_epochs: int): #max_epochs is the total number of epochs to be run
		"""
		Runs the epoch iteration consisting of:
		1) Training.
		2) Optional Validation.
		3) Checking Patience Logic.

		If model obtained best training or validation loss, the model is saved.
		"""
		postfix = {}

		epoch_iterator=(tqdm(range(self._epochs_run, max_epochs),
							initial=self._epochs_run,
							total=max_epochs,
							desc="Training Progress") if self._verbosity
						else range(self._epochs_run, max_epochs))

		for epoch in epoch_iterator:
			#Training
			self._model.train() #Set model to training mode
			train_loss=self._train()
			self._train_loss_by_epochs.append(train_loss)

			if self._writer: self._writer.add_scalar("Loss/train", train_loss, epoch)

			if self._scheduler: self._scheduler.step() #Update Scheduler post-training

			if self._verbosity:
				postfix["BestTrainLoss"] = f"{self._best_train_loss:.4e}"
				if self.val_data:
					postfix["BestValLoss"] = f"{self._best_val_loss:.4e}"

			if train_loss < self._best_train_loss:
				self._best_patience_loss=self._best_train_loss
				self._best_train_loss=train_loss
				if self._best_model_save_path and self.val_data is None:
					if self._verbosity: postfix["Best model saved at epoch"] = f"{epoch+1}"
					self._save_best_model()
			
			#Optional Validation (if validation dataloader provided)
			if self.val_data:
				self._model.eval() #Set model to inference mode
				with torch.no_grad(): #Gradient must not be updated
					val_loss=self._validation()

					self._val_loss_by_epochs.append(val_loss)
					if self._writer: self._writer.add_scalar("Loss/val", val_loss, epoch)

					if val_loss < self._best_val_loss:
						self._best_patience_loss=self._best_val_loss
						self._best_val_loss=val_loss
						if self._best_model_save_path:
							if self._verbosity: postfix["Best model saved at epoch"] = f"{epoch+1}"
							self._save_best_model()

			self._epochs_run+=1
			if self._epochs_run % self._n_save ==0 : self._collapse_check(epoch)
			if self.val_data is None:
				if abs(self._best_train_loss-self._best_patience_loss)<self._delta_patience:
					self._patience+=1
				else: self._patience=0
			else:
				if abs(self._best_val_loss-self._best_patience_loss)<self._delta_patience:
					self._patience+=1
				else: self._patience=0

			if self._writer: self._writer.flush()
			if self._patience >= self._max_patience:
				self.verbosity_logger.error("Early Stopping Triggered. Exiting training...")
				break
			if self._epochs_run % self._n_save ==0 and self._snapshot_path:
				if self._verbosity: postfix["Last snapshot saved at epoch"] = f"{epoch+1}"
				self._save_snapshot(epoch) #Backup in case something interrupts the program.

			if self._verbosity and hasattr(epoch_iterator, "set_postfix"):
				epoch_iterator.set_postfix(**postfix)

	def _train(self):
		"""
		Run Training Iteration

		Returns:
			train_loss: Training Loss.
		"""

		train_loss=0.
		for x, y in self.train_data:
			x, y =x.to(self._device), y.to(self._device)
			self._optimizer.zero_grad()
			outputs = self._model(x)
			loss = self._train_criterion(outputs, y)

			loss.backward()

			self._optimizer.step()

			train_loss+=loss.item()*x.size(0) #sum over batches, de-averaging by number of examples in batch

		train_loss = train_loss / len(self.train_data.dataset) #renormalize according to number of batches
		return train_loss

	def _validation(self):
		"""
		Run a validation iteration

		Returns:
			val_loss: Validation Loss.
		"""
		val_loss=0.
		for x, y in self.val_data:
			x, y =x.to(self._device), y.to(self._device)
			outputs = self._model(x)
			if self._val_criterion: loss = self._val_criterion(outputs, y) #val_criterion can be None
			else: loss = self._train_criterion(outputs, y)
			val_loss+=loss.item()*x.size(0)

		val_loss = val_loss / len(self.val_data.dataset)

		return val_loss

	def _collapse_check(self, epoch):
		"""
		Run an inference iteration to check for model collapse
		"""
		batch_check_result = []

		for x, _ in self.train_data:
			x=x.to(self._device)
			#Check if batch-wise and channel-wise variance is below threshold
			if torch.all(self._model(x).var(dim=tuple(range(2,len(x.shape))), correction=0) < 1e-12):
				batch_check_result.append(True)
			else: batch_check_result.append(False)

		# If variance is effectively zero channel-wise dimensions in the aggregate batch, the model is a constant predictor
		if all(batch_check_result):
			raise ModelCollapseError(f"Model collapsed at epoch {epoch+1}")
	
	#Test method to get test score
	def test(self, test_criterion: torch.nn.modules.loss._Loss, test_dataloader: torch.utils.data.DataLoader):
		"""
		Evaluate model performance on Test Set.

		Args:
			test_criterion: Criterion used to evaluate Test Performance.
			test_dataloader: DataLoader containing the Test Set.
		Returns:
			test_loss: Test loss.
		"""
		test_loss=0.
		self._model.to('cpu')
		self._model.eval()
		with torch.no_grad():
			for x, y in test_dataloader:
				x, y =x.to('cpu'), y.to('cpu')
				outputs = self._model(x)
				loss = test_criterion(outputs, y)
				test_loss+=loss.item()*x.size(0)

			test_loss = test_loss / len(test_dataloader.dataset)
			if self._writer: self._writer.add_scalar("Loss/test", test_loss)
			self._model.to(self._device)
			if self._writer: self._writer.flush()
		return test_loss
	
	
	def infer(self, input_data: torch.tensor):
		"""
		Runs model to make predictions
		Args:
			input_data: Input data Tensor
		Returns:
			output: Tensor output by model
		"""
		input_data = input_data.to('cpu')
		self._model.to('cpu')
		self._model.eval()
		with torch.no_grad():
			outputs = self._model(input_data)

		return outputs

	#Methods to plot data
	def plot_train_loss_by_epochs(self, title='Training Loss by Epochs', xlabel='Epoch', ylabel='Loss', color=None, filepath='train_loss_by_epochs.png') -> None:
		"""
		Generates and saves a pyplot plot containing Training Loss by epoch:
		Args:
			title: Title on top of the figure.
			xlabel: Label on x-axis.
			ylavel: Label on y-axis.
			color: Color of the line.
			filepath: Plot save path.
		"""

		base_name, extension = os.path.splitext(filepath)

		fig, ax = plt.subplots()
		ax.plot(np.arange(1, len(self._train_loss_by_epochs)+1), self._train_loss_by_epochs, color=color)
		ax.set_title(f"{title} with Loss Transient")
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.grid(color='black', linestyle='--', linewidth=1, alpha=0.25)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		fig.savefig(f"{base_name}_w_transient{extension}")

		fig, ax = plt.subplots()
		ax.plot(np.arange(6, len(self._train_loss_by_epochs)+1), self._train_loss_by_epochs[5:], color=color)
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.grid(color='black', linestyle='--', linewidth=1, alpha=0.25)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		fig.savefig(f"{base_name}_wo_transient{extension}")
	
	def plot_val_loss_by_epochs(self, title='Validation Loss by Epochs', xlabel='Epoch', ylabel='Loss', color=None, filepath='val_loss_by_epochs.png') -> None:
		"""
		Generates and saves a pyplot plot containing Validation Loss by epoch:
		Args:
			title: Title on top of the figure.
			xlabel: Label on x-axis.
			ylavel: Label on y-axis.
			color: Color of the line.
			filepath: Plot save path.
		"""

		base_name, extension = os.path.splitext(filepath)

		fig, ax = plt.subplots()
		ax.plot(np.arange(1, len(self._val_loss_by_epochs)+1), self._val_loss_by_epochs, color=color)
		ax.set_title(f"{title} with Loss Transient")
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.grid(color='black', linestyle='--', linewidth=1, alpha=0.25)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		fig.savefig(f"{base_name}_w_transient{extension}")

		fig, ax = plt.subplots()
		ax.plot(np.arange(6, len(self._val_loss_by_epochs)+1), self._val_loss_by_epochs[5:], color=color)
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.grid(color='black', linestyle='--', linewidth=1, alpha=0.25)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		fig.savefig(f"{base_name}_w_transient{extension}")