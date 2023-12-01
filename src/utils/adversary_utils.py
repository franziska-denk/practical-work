from typing import Tuple
import torch
import torch.nn as nn

def get_grad(model: nn.Module) -> bool:
    grad = False
    for param in model.parameters():
        grad = param.requires_grad
        break
    return grad

def get_adversary_param(opt):
    if opt.modification in ["dr", "dnr"]:
        adversary_param = {"steps": 1000,
                           "epsilon": 1e5,
                           "alpha": 0.1}
        
    elif opt.modification in ["drand", "ddet"]:
        adversary_param = {"steps": 100,
                           "epsilon": 5,
                           "alpha": 0.1}
    if opt.steps:
        adversary_param["steps"] = opt.steps
    if opt.epsilon:
        adversary_param["epsilon"] = opt.epsilon
    if opt.alpha:
        adversary_param["alpha"] = opt.alpha
        
    return adversary_param

def adv_attack(batch: Tuple[torch.Tensor, torch.Tensor],
            network: nn.Module,
            loss_func: nn.Module,
            target: torch.Tensor = None,
            epsilon: float = .1,
            alpha: float = .01,
            steps: int = 10,
            device: str = "cuda",
            clip_adv_input: bool = False):
    
    # safety measures to avoid unwanted gradient flow
    net_grad = get_grad(network)
    network.eval()
    network.requires_grad_(False)

    x_nat = batch[0].clone().detach().to(device)
    if len(x_nat.shape) == 3:
        x_nat = x_nat.unsqueeze(0)
    y = target if target is not None else batch[1]
    y = y.clone().detach().to(device)

    delta = torch.zeros_like(x_nat)
    for _ in range(steps):
        with torch.enable_grad():
            delta, x_nat = delta.clone().detach(), x_nat.clone().detach()
            x_adv = x_nat + delta
            if clip_adv_input:
                # used for dr/dnr
                x_adv = torch.clamp(x_adv, min=0, max=1)
            x_adv = x_adv.requires_grad_(True)
            output = network.forward(x_adv)
            loss = loss_func(output, y)
            loss.backward()

            with torch.no_grad():
                # normalize gradient for fixed distance in l2-norm per step
                l2_norm = x_adv.grad.view(x_adv.shape[0], -1).norm(2, dim=-1) + 1e-10
                norm_grad = x_adv.grad / l2_norm.view(-1, 1, 1, 1) 

                if target is not None:
                    x_adv = x_adv - alpha*norm_grad.clone().detach() # descent if in direction of target
                else:
                    x_adv = x_adv + alpha*norm_grad.clone().detach() # else ascent
                delta = (x_adv - x_nat).clone().detach()

                # stay in l2 norm epsilon ball around original image
                # delta * 1 if epsilon > norm, delta/norm*epsilon if norm > epsilon
                normalized_delta = delta.view(delta.shape[0], -1).norm(2, dim=-1).view(-1, 1, 1, 1)
                delta = epsilon*delta/(torch.max(normalized_delta, torch.tensor(epsilon)) + 1e-10)
        
    x_adv = x_nat + delta
    x_adv = torch.clamp(x_adv, min=0, max=1)
    network.requires_grad_(net_grad)

    return x_adv.to("cpu"), y