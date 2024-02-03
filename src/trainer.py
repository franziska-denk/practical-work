from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.adversary_utils import adv_attack

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

class Trainer:
    """Class to handle training and testing for one epoch.
    """
    
    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler = None):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer

        if scheduler:
            self.scheduler = scheduler

    def set_model(self,
                  model: nn.Module) -> None:
        """Update model used in trainer.


        Args:
            model (nn.Module): updated model
        """
        self.model = model
    
    @torch.enable_grad()
    def train(self,
              train_loader: DataLoader,
              adversary_param: dict = None) -> torch.Tensor:
        """Perform one epoch of training.

        Args:
            train_loader (DataLoader): training data loader
            adversary_param (dict, optional): if provided, adversarial training is performed. 
                                              else standard training is performed. Defaults to None.

        Returns:
            torch.Tensor: epoch loss (sum of loss over all batches)
        """
        
        self.model.train()
        self.model.requires_grad_(True)
        epoch_loss = 0

        for i, data in enumerate(train_loader):
            x = data[0].to(self.device)
            y = data[1].to(self.device)
            pred = self.model(x)

            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss += loss.item()        

            if adversary_param is not None:
                adv_data = adv_attack(data, self.model, nn.CrossEntropyLoss(), **adversary_param)
                self.model.train()
                x = adv_data[0].clone().detach().requires_grad_(False).to(self.device)
                y = adv_data[1].to(self.device)

                # make sure that no gradient can flow somewhere
                self.optimizer.zero_grad()
                self.model.zero_grad()

                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
        
        if self.scheduler:
            self.scheduler.step()

        return epoch_loss
    
    @torch.no_grad()
    def test(self,
             test_loader: DataLoader,
             adversary_param: dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one testing round.

        Args:
            test_loader (DataLoader): test data loader
            adversary_param (dict, optional): if provided, adversarial testing is performed.
                                              else standard testing. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: tuple containing test loss (sum over all losses per batch) and test accuracy [0, 1]
        """
        
        self.model.eval()
        self.model.requires_grad_(False)
        test_loss = 0; total = 0; correct = 0;
        
        for i, data in enumerate(test_loader):
            if adversary_param is not None:
                data = adv_attack(data, self.model, nn.CrossEntropyLoss(), **adversary_param)
            x = data[0].clone().detach().requires_grad_(False).to(self.device)
            y = data[1].to(self.device)
            out = self.model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            test_loss += self.criterion(out, y)
        
        return (test_loss, correct/total)