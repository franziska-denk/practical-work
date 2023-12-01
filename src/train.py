"""
Train CIFAR10 classification model in a standard and robust way.

Usage:
    * (standard training with standard dataset D):
        python train.py --wandb_name <run_name>

    * (standard training with robust dataset D_r):
        python train.py --wandb_name <run_name> --dataset_path data\d_r

    * (robust train with standard dataset D):
        python train.py --wandb_name <run_name> --include_adversary True

"""

import argparse
from datetime import datetime
import os
import sys
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import wandb

from utils.data_utils import get_cifar10_data
from trainer import Trainer
# from models.resnet_56 import resnet56

torch.manual_seed(0)

def init_weights_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        m.bias.data.fill_(0)

def init_weights_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        m.bias.data.fill_(0)

def get_time():
    return datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False, help='run in debug mode')
    parser.add_argument('--dataset_path', type=str, default=None, help='dataset that should be used. if not provided, standard dataset will be used')
    parser.add_argument('--model_name', type=str, default=None, help='model name under which checkpoint is saved')
    parser.add_argument('--ckpt_path', type=str, default="./ckpts/", help='path where checkpoint will be saved')
    parser.add_argument('--wandb_project', type=str, default='practical_work_v2', help='wandb project to be logged under')
    parser.add_argument('--wandb_name', type=str, default=None, help='name of wandb run')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size used in training')
    parser.add_argument('--n_epochs', type=int, default=70, help='number of epochs to perform training')
    parser.add_argument('--start_lr', type=float, default=0.1, help='starting learning rate to be used')
    parser.add_argument('--constant_lr', type=bool, default=False, help='True if learning rate drop should not be included')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='momentum value for SGD')
    parser.add_argument('--no_augment', type=bool, default=False, help='True if training data should not be augmented')
    parser.add_argument('--include_adversary', type=bool, default=False, help='perform adversarial training if true, else standard training')
    parser.add_argument('--adversary_steps', type=int, default=7, help='number of steps used in PGD')
    parser.add_argument('--adversary_epsilon', type=float, default=0.5, help='epsilon used in PGD')
    parser.add_argument('--adversary_alpha', type=float, default=0.1, help='alpha used in PGD')
    parser.add_argument('--resnet101', type=bool, default=False, help='True if Resnet101 should be used, else Resnet50')
    parser.add_argument('--classes', type=list, default=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck'], help='class list translating index/numerical class in string')
    try:
        return parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)
        
def main(opt):
    # load model & initialize
    if opt.resnet101:
        model = torchvision.models.resnet101()
        print("Loading Resnet101")
    else:
        model = torchvision.models.resnet50()
        print("Loading Resnet50")
    model.fc = nn.Linear(2048, len(opt.classes))
    # model = resnet56()
    model.apply(init_weights_uniform); # kaiming normal lead to worse performance
 
    now = datetime.now().strftime('%d-%m-%Y_%H-%M')
    if opt.model_name:
        model_name = opt.model_name
    elif opt.include_adversary:
        model_name = "robust_model"
    else:
        model_name = "model_" + now

    # define optimizers & schedulers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.start_lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = MultiStepLR(optimizer, [60])

    # get data
    data = get_cifar10_data(opt.batch_size, img_size=32, data_path=opt.dataset_path, augment=not opt.no_augment, subsample=opt.debug)
    _, train_loader = data["train"]
    _, val_loader = data["val"]
    _, test_loader = data["test"]


    # configure logging
    cfg = vars(opt)
    cfg["model_name"] = model_name
    run = wandb.init(project=opt.wandb_project,
               name=opt.wandb_name,
               config = cfg)
    folder_path = os.path.join(opt.ckpt_path, model_name)
    save_folder = os.makedirs(folder_path, exist_ok=True)
    best_path = os.path.join(folder_path, "best.pt")
    last_path = os.path.join(folder_path, "last.pt")
    best_adv_path = os.path.join(folder_path, "best_adv.pt")
    wandb.watch(model)
    best_model = wandb.Artifact(f"model_{opt.wandb_name}", type="model", description=f"trained model: {model_name}")
    best_adv_model = wandb.Artifact(f"adv_model_{opt.wandb_name}", type="model", description=f"trained model: {model_name}")

    # set up trainer
    trainer = Trainer(model, criterion, optimizer, scheduler) if not opt.constant_lr else Trainer(model, criterion, optimizer)
    adversary_param = {"steps": opt.adversary_steps,
                       "epsilon": opt.adversary_epsilon,
                       "alpha": opt.adversary_alpha}
    best_val_acc = 0
    best_val_adv_acc = 0
    print(vars(opt))
    print("Initialized everything, starting training...")

    # start training
    for epoch in range(opt.n_epochs):
        start = time.time()
        metrics = dict()

        # standard (and optional adversarial) training
        train_loss = trainer.train(train_loader, adversary_param if opt.include_adversary else None)
        metrics["train loss"] = train_loss/len(train_loader)

        # standard validation
        val_loss, val_acc = trainer.test(val_loader)
        metrics["validation accuracy"] = val_acc
        metrics["validation loss"] = val_loss/len(val_loader)

        torch.save(model.state_dict(), last_path)

        # perform adversarial validation if adversarial training is performed
        if opt.include_adversary:
            val_adv_loss, val_adv_acc = trainer.test(val_loader, adversary_param)
            metrics["adversarial validation accuracy"] = val_adv_acc
            metrics["adversarial validation loss"] = val_adv_loss/len(val_loader)
            # save best adversarial model
            if val_adv_acc > best_val_adv_acc: 
                torch.save(model.state_dict(), best_adv_path)
                best_val_adv_acc = val_adv_acc
                print(f"{get_time()}: saved new best robust model in epoch {epoch}")

        # save best model based on standard acc.
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), best_path)
            best_val_acc = val_acc
            print(f"{get_time()}: saved new best model in epoch {epoch}")
            
        run.log(metrics)
        end = time.time()
        s_time = end-start
        print(f"{get_time()}: Finished epoch {epoch} in {s_time//60:.0f} m {s_time%60:.0f} sec")
        
    # log best model to W&B
    best_model.add_file(best_path)
    run.log_artifact(best_model)
    if opt.include_adversary:
        best_adv_model.add_file(best_adv_path)
        run.log_artifact(best_adv_model)

    # load best model weights and test
    model.load_state_dict(torch.load(best_path))
    trainer.set_model(model)
    _, test_acc = trainer.test(test_loader)
    _, test_adv_acc = trainer.test(test_loader, adversary_param)
    wandb.log({"test accuracy": test_acc,
               "test adversarial accuracy": test_adv_acc})
    wandb.finish()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)