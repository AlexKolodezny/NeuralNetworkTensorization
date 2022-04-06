from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
import random
from functools import reduce
import operator
from torch import Tensor
from typing import List
import math
from torch.nn.init import _no_grad_normal_
from torchvision.datasets import CIFAR10
from torchvision import transforms as T


def mul(iterative):
    return reduce(operator.mul, iterative)


def xavier_normal(tensor: Tensor, in_dim: List[int], out_dim: List[int]=None, gain: float=1.0)->Tensor:
    fan_in = mul([dim for i, dim in enumerate(tensor.shape) if i in in_dim])
    if out_dim is None:
        fan_out = mul([dim for i, dim in enumerate(tensor.shape) if i not in in_dim])
    else:
        fan_out = mul([dim for i, dim in enumerate(tensor.shape) if i in out_dim])
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0., std)


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, train_dataloader, criterion, optimizer, device="cuda:0"):
    model.train()
    losses = []
    sizes = []
    for images, labels in tqdm(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        loss = criterion(pred, labels)
        loss.backward()
        losses.append(float(loss) * len(images))
        sizes.append(len(images))
        optimizer.step()
        optimizer.zero_grad()
    return losses, sizes


def predict(model, val_dataloader, criterion, device="cuda:0"):
    model.eval()
    losses = []
    labels = []
    preds = []
    with torch.no_grad():
      for images, label in val_dataloader:
          label = label.to(device)
          images = images.to(device)
          logit = model(images)
          losses.append(float(criterion(logit, label)) * len(images))
          preds.append(logit.argmax(dim=1))
          labels.append(label)
    return losses, torch.cat(preds), torch.cat(labels)


def train(model, train_dataloader, val_dataloader, criterion, optimizer, device="cuda:0", n_epochs=10, scheduler=None, plot=False, start_epoch=0, filename=None):
    accuracies = []
    for epoch in range(start_epoch, n_epochs):
        train_losses, sizes = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        # if plot:
        #     plt.plot(np.array(train_losses) / np.array(sizes))
        #     plt.show()
        val_losses, val_pred, true_pred = predict(model, val_dataloader, criterion, device)
        val_accuracy = accuracy_score(true_pred.to("cpu"), val_pred.to("cpu"))
        accuracies.append(val_accuracy)
        if scheduler is not None:
            scheduler.step()
        print("Epoch: {}, Train loss: {}, Validation loss: {}, Validation accuracy: {}"\
              .format(epoch, sum(train_losses) / len(train_dataloader.dataset), sum(val_losses) / len(val_dataloader.dataset), val_accuracy))
        if filename is not None:
            state = {
                "model": model.state_dict(),
                "epoch": epoch,
            }
            path = "./models/" + filename + ".pt"
            torch.save(state, path)
    return accuracies


def create_dataset(reflect=False):
    train_transform = T.Compose([
        T.RandomCrop((32, 32), padding=4, padding_mode="reflect" if reflect else "constant"),
        T.RandomHorizontalFlip(0.5),
        # T.ColorJitter(contrast=0.25),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
    ])

    train_dataset = CIFAR10("./data/", download=True, train=True, transform=train_transform)
    val_dataset = CIFAR10("./data/", download=True, train=False, transform=val_transform)
    return train_dataset, val_dataset
