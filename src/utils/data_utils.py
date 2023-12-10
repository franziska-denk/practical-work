import os
import math
from typing import Tuple, Union
import pickle

import torch
from PIL import Image
from torchvision import (transforms,
                         datasets)

from torch.utils.data import (random_split,
                              DataLoader,
                              Dataset,
                              Subset)

def get_cifar10_data(BATCH_SIZE: int = 64,
                     img_size: int = 32,
                     data_path: str = None,
                     augment: bool = True,
                     subsample: bool = False,
                     train_shuffle: bool = True) -> dict[str, 
                                                  Union[torch.Tensor, Tuple]]:
    
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar 

    test_transform_list = [transforms.ToTensor()]#,
                           #transforms.Resize((img_size,img_size))]
    
    if augment:
        train_transform_list = test_transform_list + [#transforms.RandomVerticalFlip(p=0.2),
                                            transforms.RandomResizedCrop(size=img_size, scale=(0.6, 1)),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ColorJitter(.25,.25,.25),
                                            transforms.RandomRotation(2)]#,
                                            #transforms.RandomResizedCrop((64,64), scale=(0.6, 1))]
    else:
        train_transform_list = test_transform_list
        
    test_transform = transforms.Compose(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform);
    train_set, val_set = random_split(train_set, [math.floor(len(train_set)*4/5), math.ceil(len(train_set)*1/5)], generator=torch.Generator().manual_seed(42)) # https://stackoverflow.com/questions/55820303/fixing-the-seed-for-torch-random-split

    if data_path:
        # load modified dataset
        # overwrite train data, evaluate on standard data
        print("loading modified dataset ...")
        train_set = ModifiedCifar(os.path.join(data_path, "img"), os.path.join(data_path, "labels.pkl"), train_transform)
    
    test_set = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform);

    if subsample:
        train_set = Subset(train_set, range(int(len(train_set)*0.25)))
        val_set = Subset(train_set, range(int(len(val_set)*0.25)))
        test_set = Subset(test_set, range(int(len(test_set)*0.25)))

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=train_shuffle, drop_last=True)
    val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=True, drop_last=False)

    return {"train": (train_set, train_loader),
            "val": (val_set, val_loader),
            "test": (test_set, test_loader),
            "classes": classes}

class ModifiedCifar(Dataset):
    def __init__(self,
                 img_dir: str,
                 label_path: str,
                 transform=None):
        
        self.img_dir = img_dir
        with open(label_path, "rb") as file:
            self.labels = pickle.load(file)
        self.transform = transform
    
    def __len__(self):
       return len(self.labels)
    
    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.img_dir, f"{idx}.jpg")
        img = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        
        return img, label