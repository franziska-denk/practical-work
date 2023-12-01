"""
Test CIFAR10 classification model in a standard and robust way.

Usage:
    * (standard testing):
        python test.py --model_name <model_name>
    * (test with non-default attack parameters):
        python test.py --model_name <model_name> --adversary_steps 10 --adversary_epsilon 0.1 --adversary_alpha 0.01

"""

import argparse
import os
import sys

import torch
import torchvision
import torch.nn as nn


from utils.data_utils import get_cifar10_data
from trainer import Trainer

torch.manual_seed(0)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='dataset that should be used. if not provided, standard dataset will be used')
    parser.add_argument('--model_name', type=str, default=None, help="Name of model checkpoint to be tested")
    parser.add_argument('--ckpt_path', type=str, default="./ckpts/", help='path where checkpoint is saved')
    parser.add_argument('--adversary_steps', type=int, default=7, help='number of steps used in PGD')
    parser.add_argument('--adversary_epsilon', type=float, default=0.5, help='epsilon used in PGD')
    parser.add_argument('--adversary_alpha', type=float, default=0.1, help='alpha used in PGD')
    parser.add_argument('--classes', type=list, default=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck'], help='class list translating index/numerical class in string')
    try:
        return parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)

def test(model_name: str,
         ckptpath: str = "./ckpts/",
         adversary_steps: int = 7,
         adversary_epsilon: float = 0.5,
         adversary_alpha = 0.1,
         classes: list=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck'],
         dataset=None):
    
    model = torchvision.models.resnet50()
    model.fc = nn.Linear(2048, len(classes))
    model.load_state_dict(torch.load(os.path.join(ckptpath, model_name)))

    criterion = nn.CrossEntropyLoss()

    data = get_cifar10_data(128, img_size=32, augment=False)
    _, test_loader = data["test"]

    trainer = Trainer(model, criterion)
    adversary_param = {"steps": adversary_steps,
                       "epsilon": adversary_epsilon,
                       "alpha": adversary_alpha}

    print("Initialized everything, starting testing...")
    
    _, test_acc = trainer.test(test_loader)
    _, test_adv_acc = trainer.test(test_loader, adversary_param)
    print(f"Standard test accuracy: {test_acc:.4}")
    print(f"Adversarial test accuracy: {test_adv_acc:.4}")
        
def main(opt):
    print(vars(opt))
    test(opt.model_name, opt.ckpt_path, opt.adversary_steps, opt.adversary_epsilon, opt.adversary_alpha, opt.classes)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)