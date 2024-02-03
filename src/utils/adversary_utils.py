from typing import Tuple, Optional
import torch
import torch.nn as nn
from captum.robust import PGD

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

def get_grad(model: nn.Module) -> bool:
    grad = False
    for param in model.parameters():
        grad = param.requires_grad
        break
    return grad

def get_adversary_param(opt):
    if opt.modification in ["dr", "dnr"]:
        adversary_param = {"steps": 1000,
                           "epsilon": None,
                           "alpha": 0.1}
        
    elif opt.modification in ["drand", "ddet"]:
        adversary_param = {"steps": 100,
                           "epsilon": 1,
                           "alpha": 0.1}
    if opt.steps:
        adversary_param["steps"] = opt.steps
    if opt.epsilon:
        adversary_param["epsilon"] = opt.epsilon
    if opt.alpha:
        adversary_param["alpha"] = opt.alpha
        
    return adversary_param

def random_init_delta_l2(original_sample: torch.Tensor,
                      epsilon: float):
    bs = original_sample.shape[0]
    delta = torch.empty_like(original_sample).normal_()
    norm_delta = delta.view(bs, -1).norm(p=2, dim=1).view(bs, 1, 1, 1)
    r = torch.zeros_like(norm_delta).uniform_(0, 1)
    return delta * r / norm_delta * epsilon

def random_init_delta_linf(original_sample: torch.Tensor,
                           epsilon: float):
    return torch.empty_like(original_sample).uniform_(-epsilon, epsilon)

def get_delta(original_sample: torch.Tensor,
              epsilon: float,
              random_start: bool,
              norm: str):
    if random_start and norm=="linf":
        return random_init_delta_l2(original_sample, epsilon)
    elif random_start and norm=="l2":
        return random_init_delta_linf(original_sample, epsilon)
    else:
        return torch.zeros_like(original_sample)

def adv_attack(batch: Tuple[torch.Tensor, torch.Tensor],
            network: nn.Module,
            loss_func: nn.Module,
            target: torch.Tensor = None,
            epsilon: float = .1,
            alpha: float = .01,
            steps: int = 10,
            device: str = "cuda",
            clip_adv_input: bool = False,
            norm: str = "l2",
            random_start: bool = False):
    
    # safety measures to avoid unwanted gradient flow
    net_grad = get_grad(network)
    network.eval()
    network.requires_grad_(False)

    x_nat = batch[0].clone().detach().to(device)
    if len(x_nat.shape) == 3:
        x_nat = x_nat.unsqueeze(0)
    y = target if target is not None else batch[1]
    y = y.clone().detach().to(device)

    delta = get_delta(x_nat, epsilon, random_start, norm=norm)
    for _ in range(steps):
        with torch.enable_grad():
            delta, x_nat = delta.clone().detach(), x_nat.clone().detach()
            x_adv = x_nat + delta
            if clip_adv_input:
                # used for dr/dnr
                x_adv = torch.clamp(x_adv, min=0, max=1)
            x_adv.requires_grad = True
            output = network.forward(x_adv)
            loss = loss_func(output, y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            #grad = torch.sign(grad)

            with torch.no_grad():
                if norm == "l2":
                    # normalize gradient for fixed distance in l2-norm per step
                    l2_norm = grad.view(x_adv.shape[0], -1).norm(2, dim=1) + 1e-10
                    norm_grad = grad/ l2_norm.view(-1, 1, 1, 1)
                    #norm_grad = torch.sign(norm_grad)

                    multipl = (-1) if target is not None else 1 # descent if in direction of target else ascent
                    x_adv = x_adv + multipl*alpha*norm_grad
                    delta = (x_adv - x_nat).clone().detach()

                    # stay in l2 norm epsilon ball around original image
                    # delta * 1 if epsilon > norm, delta/norm*epsilon if norm > epsilon
                    if epsilon:
                        # delta_norms = torch.norm(delta.view(x_adv.shape[0], -1), p=2, dim=1)
                        # factor = epsilon / delta_norms
                        # factor = torch.min(factor, torch.ones_like(delta_norms))
                        # delta = delta * factor.view(-1, 1, 1, 1)
                        delta = delta.renorm(p=2, dim=0, maxnorm=epsilon)

                elif norm == "linf":
                    # normalize gradient for fixed distance in linfty-norm per step
                    norm_grad = torch.sign(grad)

                    multipl = (-1) if target is not None else 1 # descent if in direction of target else ascent
                    x_adv = x_adv + multipl*alpha*norm_grad
                    delta = (x_adv - x_nat).clone().detach()

                    # stay in linfty norm epsilon ball around original image
                    # delta * 1 if epsilon > norm, delta/norm*epsilon if norm > epsilon
                    if epsilon:
                        delta = torch.clamp(delta, -epsilon, epsilon)
                else:
                    raise ValueError("Norm not supported.")
        
    x_adv = x_nat + delta
    x_adv = torch.clamp(x_adv, min=0, max=1)
    network.requires_grad_(net_grad)

    return x_adv.to("cpu"), y

class CaptumAttack:
    def __init__(self,
                network: nn.Module,
                loss_func: nn.Module,
                epsilon: float = .1,
                alpha: float = .01,
                steps: int = 10,
                device: str = "cuda"):
        self.pgd = PGD(network, loss_func, lower_bound=0, upper_bound=1)
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.device = device
    
    def perturb(self,
                batch: Tuple[torch.Tensor, torch.Tensor],
                target: Optional[torch.Tensor] = None,
                epsilon: Optional[float] = None,
                alpha: Optional[float] = None,
                steps: Optional[float] = None):
        
        eps = self.epsilon if not epsilon else epsilon
        alpha = self.alpha if not alpha else alpha
        steps = self.steps if not steps else steps
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        if target:
            return self.pgd.perturb(x, radius=eps, step_size=alpha, step_num=steps, target=target, targeted=True)
        else:
            return self.pgd.perturb(x, radius=eps, step_size=alpha, step_num=steps, target=y)