from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


def vis_data(dataset: Dataset,
             classes: Tuple[str],
             n_samples: int = 3):
    
    fig, axs = plt.subplots(1, n_samples, figsize=((3*n_samples, 5)))
    for i in range(n_samples):
        img_whc = dataset[i][0].numpy().transpose(1,2,0) #/ 2 + 0.5
        label = dataset[i][1]
        axs[i].imshow(img_whc)
        axs[i].set_title(f"Label: {classes[label]}")
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    return fig

def vis_pred(sample: Tuple[torch.Tensor, int],
             model: nn.Module,
             classes: Tuple[str],
             device: str = "cuda"):
    model.eval()
    _, pred = model(sample[0].unsqueeze(0).to(device)).max(1)
    img_whc = sample[0].numpy().transpose(1,2,0) #/ 2 + 0.5
    label = sample[1]
    plt.imshow(img_whc)
    plt.title(f"Predicted label: {classes[pred]}. True label: {classes[label]}")
    plt.axis("off")

    return plt.show()