import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader

class BaseDLFrameworkDDP:
	def __init__(self,
			    model: torch.nn.Module,
			    train_data: DataLoader,
			    optimizer: torch.optim.Optimizer,
			    criterion: torch.nn.modules.loss._Loss,
			    scheduler: torch.optim.lr_scheduler,
			    snapshot_path: str,
			    model_init_path: str,
			    save_every_epoch: int = 5) -> None:
		self._gpu_id = int(os.environ["LOCAL_RANK"])
		self._model = model.to(self._gpu_id)
		self.train_data = train_data
		self._optimizer = optimizer
		self._criterion = criterion
		self._scheduler=scheduler
		self.epochs_run = 0
		self.snapshot_path = snapshot_path
		self._n_save=save_every_epoch #Save snapshot every n_save epochs
		if os.path.exists(snapshot_path):
			print("Loading snapshot")
			self._load_snapshot(snapshot_path)
		elif os.path.exists(model_init_path):
			print("Loading Initialized model")
			self._load_init_model(model_init_path)
		self._model = DDP(self._model, device_ids=[self._gpu_id])
		self._train_loss_by_epochs = []
		self._val_loss_by_epochs = []
		self.best_valid_loss=float('inf')

	#Training load and save methods
	def _save_snapshot(self, epoch): ##Backup the training in case of DDP errors, so training can be resumed instead of a reset
		snapshot = {
		  "MODEL_STATE": self._model.module.state_dict(),
		  "EPOCHS_RUN": epoch,
		  "TRAIN_LOSS_EPOCHS": self._train_loss_by_epochs
		}
		if not os.path.exists(os.path.abspath(os.path.dirname(self._snapshot_path))):
			os.mkdir(os.path.abspath(os.path.dirname(self._snapshot_path)))
		torch.save(snapshot, self._snapshot_path)
		print(f"Epoch {epoch+1} | Training snapshot saved at {self._snapshot_path}")

	def _load_snapshot(self): ##Load the backup at declaration of Trainer class in main()
		loc = f"cuda:{self._gpu_id}"
		snapshot = torch.load(self.snapshot_path, map_location=loc)
		self._model.load_state_dict(snapshot["MODEL_STATE"])
		self._epochs_run = snapshot["EPOCHS_RUN"]
		self._train_loss_by_epochs = snapshot["TRAIN_LOSS_EPOCHS"]
		print(f"Resuming training from snapshot saved at Epoch {self._epochs_run}")

	#Methods to get and load information on the class
	def _load_init_model(self, model_init_path): ##Load the backup at declaration of Trainer class in main()
		loc = f"cuda:{self._gpu_id}"
		init_weights = torch.load(model_init_path, map_location=loc)
		self.model.load_state_dict(init_weights["MODEL_STATE"])
		print(f"Initialized the weights of the model")

	def _save_best_model(self): #Save the model which performed the best
		best_model={
		    "MODEL_STATE": self._model.module.state_dict(),
		    "MODEL_ARCH": str(self._get_model),
		    "BEST_LOSS": self._best_train_loss,
		    "OPTIM_HYPERPARM": self._get_optim_hp()
			}
		torch.save(best_model, self._best_model_save_path)

	def _get_optim_hp(self):
		if self._gpu_id==0:
			for param_group in self._optimizer.param_groups:
				return {key: value for key, value in param_group.items() if key != "params"}

	def _get_model(self):
		return self._model #Can be used inside print()

	#Methods for training
	def run_epochs(self, max_epochs: int): #max_epochs is the total number of epochs to be run
		for epoch in range(self._epochs_run, max_epochs):
			self._model.train()
			b_sz = len(next(iter(self.train_data)))
			print(f"[GPU{self._gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
			train_loss=self._train()
			dist.barrier() #Synchronize all ranks before saving
			if self._gpu_id == 0:
				if train_loss < self._best_train_loss and self._epochs_run+1 >5:
					self._best_train_loss=train_loss
					print(f"Saving best model at Epoch {self._epochs_run+1}")
					self._save_best_model()
				self._epochs_run+=1
				if epoch % self._n_save == 0: self._save_snapshot(epoch) ##Backup of training, in case something interrupts the program. Best to run after validation, if present
			dist.barrier() #Synchronize all ranks after saving

	def _train(self):
		train_loss=0.

		for x, y in self.train_data:
			x, y =x.to(self._gpu_id), y.to(self._gpu_id)
			self._optimizer.zero_grad()
			outputs = self._model(x)
			loss = self._criterion(outputs, y)
			loss.backward()

			self._optimizer.step()

			train_loss+=loss.item()*x.size(0)

		train_loss = train_loss / len(self.train_data.dataset)
		if self._gpu_id==0:
			self._train_loss_by_epochs.append(train_loss)
		self._scheduler.step()
		return train_loss
		

	#Methods to plot data
	def plot_train_loss_by_epochs(self, title=None, xlabel=None, ylabel=None, label=None, color=None, filename='test_loss.png'):
		if self._gpu_id==0:
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
