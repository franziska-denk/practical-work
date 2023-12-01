"""
Create modified datasets

Usage:
    * (create Dr):
        python modify_data.py --modification dr --model_name <robust_model_name> --save_path ./data/d_r

    * (create Ddet)
        python modify_data.py --modification ddet --model_name <standard_model_name> --save_path ./data/d_det

"""

import argparse
import os
import pickle
import sys
from typing import Tuple

import torch
import torchvision
from torch import nn as nn
from PIL import Image

from utils.data_utils import get_cifar10_data
from utils.adversary_utils import (adv_attack,
                                   get_adversary_param)

BATCH_SIZE = 128

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modification', type=str, default='dr', help='which modified dataset should be created: dr, dnr, ddet, drand')
    parser.add_argument('--save_path', type=str, default="./data/d_r", help='path under which data will be saved')
    parser.add_argument('--model', type=str, default='robust_model_32_25-10-2023_16-35.pt', help='model used to create modified dataset')
    parser.add_argument('--resnet101', type=bool, default=False, help='True if Resnet101 should be used, else Resnet50')
    parser.add_argument('--steps', type=int, default=None, help='steps used to create to create single sample')
    parser.add_argument('--epsilon', type=float, default=None, help='epsilon used to create samples')
    parser.add_argument('--alpha', type=float, default=None, help='alpha used in creating samples')
    try:
        return parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)

class Modifier:
    def __init__(self,
                 model: nn.Module,
                 adversary_param: dict,
                 n_classes: int = None):
        
        self.model = model
        self.adversary_param = adversary_param
        self.n_classes = n_classes
        self.mapping = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def modify_dr_dnr(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      x_r: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate D_r by using PGD to only use robust features, D_nr is constructed as controll dataset

            * D_r:  make random input x_r more similar to features of target image
                    by using robust models last layer as feature representation
                    and optimize x_r towards this feature representation
                    -> optimized x_r only contains robust features that correlate with target

            * D_nr: repeat same process as for D_r, but with standard model as feature extractor
                    -> optimized x_r contains non-robust features as well

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (target_image, target_label) towards which is optimized
            x_r (Tuple[torch.Tensor, torch.Tensor]): random input image that is optimized towards target_label

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (optimized_x_r, target_label)
        """
        x, y_target = batch
        g_x = self.model(x.to(self.device))
        mod_x, _ = adv_attack(batch=x_r,
                              model=self.model,
                              loss_func=nn.MSELoss(),
                              target=g_x,
                              clip_adv_input=True,
                              **self.adversary_param)

        return mod_x, y_target

    def modify_drand(self,
                     batch: Tuple[torch.Tensor, torch.Tensor])  -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate D_rand by using PGD to only contain non-robust features.
           Take input image and optimize it towards new random target label.
           Correlation with new target label will only be due to non-robust features, rest of image is uncorrelated w/ target label.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (original_image, original_label) where original_image will be modified

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (optimized_image, target_label)
        """
        y_target = torch.randint(low=0, high=self.n_classes-1, size=(batch[0].shape[0],))
        mod_x, y = adv_attack(batch=batch,
                              model=self.model,
                              loss_func=nn.CrossEntropyLoss(),
                              target=y_target,
                              **self.adversary_param)
        
        return mod_x, y_target
    
    def modify_ddet(self,
                    batch: Tuple[torch.Tensor, torch.Tensor])  -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate D_det by using PGD to only contain non-robust features.
           Take input image and optimize it towards new target label (t = (y+1)%n_classes), which is deterministically found based on original target.
           Correlation with new target label will only be due to non-robust features, robust features point away from new target label.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (original_image, original_label) where original_image will be modified

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (optimized_image, target_label)
        """
        x,y = batch
        y_target = (y+1) % self.n_classes
        mod_x, y = adv_attack(batch=batch,
                              model=self.model,
                              loss_func=nn.CrossEntropyLoss(),
                              target=y_target,
                              **self.adversary_param)
        
        return mod_x, y_target
        

def main(opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = get_cifar10_data(BATCH_SIZE, 32, subsample=False, augment=False)
    _, train_loader = data["train"]

    if opt.resnet101:
        model = torchvision.models.resnet101()
        print("Loading Resnet101")
    else:
        model = torchvision.models.resnet50()
        print("Loading Resnet50")
    model.fc = nn.Linear(2048, len(data["classes"]))
    #model = resnet56()
    model.to(device)

    MODEL_PATH = f"ckpts/{opt.model}"
    model.load_state_dict(torch.load(MODEL_PATH));
    model.eval();
    if opt.modification in ["dr", "dnr"]:
        model.fc = nn.Identity()
        #model.linear = nn.Identity()
        #model.pool = nn.Identity()

    adversary_param = get_adversary_param(opt)
    print(adversary_param)

    labels = dict()
    modifier = Modifier(model, n_classes=len(data["classes"]), adversary_param=adversary_param)
    os.makedirs(os.path.join(opt.save_path, "img"), exists_ok=True)

    print(f"Using modification {opt.modification} ...")
    idx = 0
    for batch in train_loader:
        
        if opt.modification in ["dr", "dnr"]: 
            x_r = next(iter(train_loader))
            mod_x, y_target = modifier.modify_dr_dnr(batch, x_r)
        elif opt.modification == "drand":
            mod_x, y_target = modifier.modify_drand(batch)
        elif opt.modification == "ddet":
            mod_x, y_target = modifier.modify_ddet(batch)
        else:
            raise ValueError("Invalid modification method.")
        
        for i in range(mod_x.shape[0]):
            im = mod_x[i].squeeze().numpy().transpose(1,2,0) * 255
            img = Image.fromarray(im.astype("uint8"))
            img.save(os.path.join(opt.save_path, "img", f"{idx+i}.jpg"))
            labels[idx+i] = y_target[i]

        print(f"done with {idx+i}")
        idx += BATCH_SIZE

    with open(os.path.join(opt.save_path, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)


if __name__=="__main__":
    opt = parse_opt()
    main(opt)