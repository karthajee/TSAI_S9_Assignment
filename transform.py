import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Custom class definition for albumentation transforms
# to make them operable with Pytorch training
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img):        
        return self.transforms(image=np.array(img))['image']

# Defining the albumentation transforms
def get_transforms(means, stds):
  
  train_transforms = Transforms(A.Compose([
      A.HorizontalFlip(p=0.5),
      A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
      A.CoarseDropout(max_holes = 1, max_height=16, 
                      max_width=16, min_holes = 1, min_height=16, 
                      min_width=16, fill_value=means, mask_fill_value = None),
      A.Normalize(mean=means, std=stds),
      ToTensorV2(),
  ]))
  test_transforms = Transforms(A.Compose([
      A.Normalize(mean=means, std=stds),
      ToTensorV2(),
  ]))
  return train_transforms, test_transforms