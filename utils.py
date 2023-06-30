import torch
import platform
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np

def get_device():
  if platform.system().lower() == 'darwin':
    use_gpu = torch.backends.mps.is_built()
    dev_name = "mps"
  elif torch.cuda.is_available():    
    dev_name = "cuda"
  else:
    dev_name = "cpu"
  device = torch.device(dev_name)
  return device

def print_summary(model, in_size=(1, 3, 32, 32)):
  print(summary(model, in_size))

def visualize(Trainer):
  train_test_diff = [np.abs(tr-te)/100 for tr, te in zip(Trainer.train_acc, Trainer.test_acc)]
  fig, axs = plt.subplots(ncols=5,figsize=(25,5));
  axs[0].plot(Trainer.train_losses);
  axs[0].set(title="Train loss", xlabel="Steps");

  axs[1].plot(Trainer.test_losses);
  axs[1].set(title="Test loss", xlabel="Epochs");

  axs[2].plot(Trainer.train_acc);
  axs[2].set(title="Train accuracy", xlabel="Steps");

  axs[3].plot(Trainer.test_acc);
  axs[3].set(title="Test accuracy", xlabel="Epochs");

  axs[4].plot(train_test_diff);
  axs[4].set(title="Train-test acc difference", xlabel="Epochs");