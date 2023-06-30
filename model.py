import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
  """
  Custom class for a convolution block
  """
  def __init__(self, in_C, out_C, p=0.1):
    super().__init__()
    self.conv2d_kwargs_1 = dict(kernel_size=3, padding="same", bias=False)
    self.conv2d_kwargs_2 = dict(kernel_size=3, padding=2, dilation=2, padding_mode="reflect", bias=False)
    block_l = []
    for i in range(3):
      subblock_l = []
      if i == 0:
        subblock_l.append(DepthConv2d(in_C, out_C, **self.conv2d_kwargs_1))
      elif i == 1:
        subblock_l.append(DepthConv2d(out_C, out_C, **self.conv2d_kwargs_1))
      else:
        subblock_l.append(DepthConv2d(out_C, out_C, **self.conv2d_kwargs_2))
      subblock_l.append(nn.BatchNorm2d(out_C))
      subblock_l.append(nn.Dropout(p))
      subblock_l.append(nn.ReLU())
      block_l.append(nn.Sequential(*subblock_l))
    self.block = nn.Sequential(*block_l)
    
  def __call__(self, x):
    i = 0
    for subblock in self.block.children():
      if i == 0:        
        x = subblock(x)         
      else:        
        x = x + subblock(x)        
      i += 1
    return x
      
class DepthConv2d(nn.Module):

  """
  Depthwise Separable Convolution Layer class
  (Inherits from nn.Module)

  """
  def __init__(self, in_C, out_C, **kwargs):
    
    """
    Initialize the Layer
    Args:
      in_C (int): Number of input channels
      out_C (int): Number of output channels
    """
    super(DepthConv2d, self).__init__()
    self.layer = nn.Sequential(
      # Decouple spatial and channel convolution
      nn.Conv2d(in_C, in_C, groups=in_C, **kwargs),
      # Recombine using standard pointwise convolution
      nn.Conv2d(in_C, out_C, kernel_size=1)
    )
    
  def __call__(self, x):
    return self.layer(x)

class Net(nn.Module):
  
  """
  Custom Neural Network class of C1C2C3C4O architecture
  outlined in class (inherits from nn.Module)
  """

  def __init__(self, C1_C=32, C2_C=64, C3_C=128, C4_C=256, p=0.1):

    """
    Initializes the Network
    Args:
      C1_C (int): Number of channels in block C1. Defaults to 32
      C2_C (int): Number of channels in block C2. Defaults to 64.
      C3_C (int): Number of channels in block C3. Defaults to 128.
      C4_C (int): Number of channels in block C4. Defaults to 256.
      p (float): Dropour probability. Defaults to 0.1.
    """
    super(Net, self).__init__()
    self.C1_C=C1_C
    self.C2_C=C2_C
    self.C3_C=C3_C
    self.C4_C=C4_C    

    self.C1 = ConvBlock(3, self.C1_C)
    self.C2 = ConvBlock(self.C1_C, self.C2_C, p)
    self.C3 = ConvBlock(self.C2_C, self.C3_C, p)
    self.C4 = ConvBlock(self.C3_C, self.C4_C, p)
    self.O = nn.Sequential(
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Conv2d(self.C4_C, 10, kernel_size=1, padding=0)
    )

  def forward(self, x):
    """
    Forward pass of the Network
    Args:
      x (Tensor): Batch of input images
    Returns:
      o (Tensor): Network prediction output
    """
    x = self.C1(x)
    x = self.C2(x)    
    x = self.C3(x)
    x = self.C4(x)
    x = self.O(x)
    o = F.log_softmax(x.squeeze())
    return o

  def save(self, dir='weights/'):
    
    """
    Saves model weights with name created as per block channels
    Args:
      dir (str): Directory to save the model weights
    """

    filepath = dir + f"{self.C1_C}_{self.C2_C}_{self.C3_C}_{self.C4_C}.pt"
    torch.save(self, filepath)

class Net_old(nn.Module):
  
  """
  Custom Neural Network class of C1C2C3C4O architecture
  outlined in class (inherits from nn.Module)
  """

  def __init__(self, C1_C=32, C2_C=64, C3_C=128, C4_C=256, p=0.1):

    """
    Initializes the Network
    Args:
      C1_C (int): Number of channels in block C1. Defaults to 32
      C2_C (int): Number of channels in block C2. Defaults to 64.
      C3_C (int): Number of channels in block C3. Defaults to 128.
      C4_C (int): Number of channels in block C4. Defaults to 256.
      p (float): Dropour probability. Defaults to 0.1.
    """
    super(Net, self).__init__()
    self.C1_C=C1_C
    self.C2_C=C2_C
    self.C3_C=C3_C
    self.C4_C=C4_C
    conv2d_kwargs_1 = dict(kernel_size=3, padding="same", bias=False)
    conv2d_kwargs_2 = dict(kernel_size=3, dilation=2, bias=False)

    self.C1 = nn.Sequential(
        # n_in: 32, s : 1, j_in: 1, r_in: 1 | n_out: 32, j_out: 1, r_out: 3
        DepthConv2d(3, C1_C, **conv2d_kwargs_1),
        nn.BatchNorm2d(C1_C),
        nn.Dropout(p),
        nn.ReLU(),
        # n_in: 32, s : 1, j_in: 1, r_in: 3 | n_out: 32, j_out: 1, r_out: 5
        DepthConv2d(C1_C, C1_C, **conv2d_kwargs_1),
        nn.BatchNorm2d(C1_C),
        nn.Dropout(p),
        nn.ReLU(),
        # n_in: 32, s : 2, j_in: 1, r_in: 5 | n_out: 16, j_out: 2, r_out: 7
        DepthConv2d(C1_C, C1_C, **conv2d_kwargs_2),
        nn.BatchNorm2d(C1_C),
        nn.Dropout(p),
        nn.ReLU(),
    )

    self.C2 = nn.Sequential(
        # n_in: 16, s : 1, j_in: 2, r_in: 7 | n_out: 16, j_out: 2, r_out: 11
        DepthConv2d(C1_C, C2_C, **conv2d_kwargs_1),
        nn.BatchNorm2d(C2_C),
        nn.Dropout(p),
        nn.ReLU(),
        # n_in: 16, s : 1, j_in: 1, r_in: 11 | n_out: 16, j_out: 2, r_out: 15
        DepthConv2d(C2_C, C2_C, **conv2d_kwargs_1),
        nn.BatchNorm2d(C2_C),
        nn.Dropout(p),
        nn.ReLU(),
        # n_in: 16, s : 2, j_in: 1, r_in: 15 | n_out: 8, j_out: 4, r_out: 19
        DepthConv2d(C2_C, C2_C, **conv2d_kwargs_2),
        nn.BatchNorm2d(C2_C),
        nn.Dropout(p),
        nn.ReLU(),
    )

    self.C3 = nn.Sequential(
        # n_in: 8, s : 1, j_in: 4, r_in: 19 | n_out: 8, j_out: 4, r_out: 27
        DepthConv2d(C2_C, C3_C, **conv2d_kwargs_1),
        nn.BatchNorm2d(C3_C),
        nn.Dropout(p),
        nn.ReLU(),
        # n_in: 8, s : 1, j_in: 4, r_in: 27 | n_out: 8, j_out: 4, r_out: 35
        DepthConv2d(C3_C, C3_C, **conv2d_kwargs_1),
        nn.BatchNorm2d(C3_C),
        nn.Dropout(p),
        nn.ReLU(),
        # n_in: 8, s : 2, j_in: 4, r_in: 35 | n_out: 4, j_out: 8, r_out: 43
        DepthConv2d(C3_C, C3_C, **conv2d_kwargs_2),
        nn.BatchNorm2d(C3_C),
        nn.Dropout(p),
        nn.ReLU(),
    )
    
    self.C4 = nn.Sequential(
        # n_in: 4, s : 1, j_in: 8, r_in: 43 | n_out: 4, j_out: 8, r_out: 59                
        DepthConv2d(C3_C, C4_C, **conv2d_kwargs_1),
        nn.BatchNorm2d(C4_C),
        nn.Dropout(p),
        nn.ReLU(),
        # n_in: 4, s : 1, j_in: 8, r_in: 59 | n_out: 4, j_out: 8, r_out: 75
        DepthConv2d(C4_C, C4_C, **conv2d_kwargs_1),        
        nn.BatchNorm2d(C4_C),
        nn.Dropout(p),
        nn.ReLU(),
        # n_in: 4, s : 2, j_in: 8, r_in: 75 | n_out: 2, j_out: 16, r_out: 91
        DepthConv2d(C4_C, C4_C, **conv2d_kwargs_2),        
        nn.BatchNorm2d(C4_C),
        nn.Dropout(p),
        nn.ReLU(),
    )

    self.O = nn.Sequential(
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Conv2d(C4_C, 10, kernel_size=1, padding=0)
    )

  def forward(self, x):
    """
    Forward pass of the Network
    Args:
      x (Tensor): Batch of input images
    Returns:
      o (Tensor): Network prediction output
    """
    x = self.C1(x)
    x = self.C2(x)
    x = self.C3(x)
    x = self.C4(x)
    x = self.O(x)
    o = F.log_softmax(x.squeeze())
    return o

  def save(self, dir='weights/'):
    
    """
    Saves model weights with name created as per block channels
    Args:
      dir (str): Directory to save the model weights
    """

    filepath = dir + f"{self.C1_C}_{self.C2_C}_{self.C3_C}_{self.C4_C}.pt"
    torch.save(self, filepath)