import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import ssl
import argparse
from src.utils import create_dataset, set_random_seed, train, predict
import tensornetwork as tn
import numpy as np
from copy import deepcopy
import json

import sys


parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", default=12345, type=int)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--filename", help="File to store checkpoints", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--full_train", default=2, type=int)
parser.add_argument("--edge_train", default=2, type=int)
parser.add_argument("--slice_gain", default=1, type=float)
parser.add_argument("--warmup", default=10, type=int)
parser.add_argument("--learning_rate", default=0.1, type=float)
parser.add_argument("--tensorization", nargs="+", type=int, default=[8, 8, 8, 8])
args = parser.parse_args()

print(sys.argv)

print("Initializing dataloader")
ssl._create_default_https_context = ssl._create_unverified_context
train_dataset, val_dataset = create_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
train_val_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=128)
print("Dataloader has initialized")

from src.networks.resnet import resnet32
from src.layers.full_linked import FullLinkedLayer

n_factors = len(args.tensorization) + 1

print("Initializing model")
set_random_seed(args.seed)
model = resnet32()
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Unflatten(1, tuple(args.tensorization)),
    FullLinkedLayer(tuple(args.tensorization), (10,), [[1] * n_factors for i in range(n_factors)]),
)
def get_layer(model):
    return model.classifier[-1]
def set_layer(model, new_layer):
    model.classifier[-1] = new_layer

start_epoch = 0

criterion = nn.CrossEntropyLoss(reduction="mean")

device = args.device if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print("Model initialized")

for core in get_layer(model).construct_network():
    print(core.name, core.shape)

print("Testing model")
all_losses, predicted_labels, true_labels = predict(model, val_dataloader, criterion, device)
assert len(predicted_labels) == len(val_dataset)
accuracy = accuracy_score(predicted_labels.to("cpu"), true_labels.to("cpu"))
print("Tests passed")

from src.greedy_tn import choose_and_increase_edge

print("Warm up model")
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=2e-4)
def learning_rate(epoch):
    if epoch < args.warmup:
        return (epoch + 1) / args.warmup
    if epoch <= 100:
        return 1
    elif epoch <= 150:
        return 0.1
    else:
        return 0.01
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate, verbose=True)
train(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    scheduler=scheduler,
    n_epochs=args.warmup,
    start_epoch=0,
    device=device)

print("Stop warm up model")

print("Choosign ranks")
def train_edge(model):
    for core in get_layer(model).construct_network():
        print(core.name, core.shape)
    optimizer = torch.optim.SGD(get_layer(model).parameters(), lr=args.learning_rate, momentum=0.9)
    # all_losses, _, _ = predict(model, train_val_dataloader, criterion, device)
    # print("Before train edge training loss: {}".format(np.sum(all_losses) / len(train_val_dataloader.dataset)))
    train(model, train_val_dataloader, val_dataloader, criterion, optimizer, scheduler=None, n_epochs=args.edge_train, device=device)
    # all_losses, _, _ = predict(model, train_val_dataloader, criterion, device)
    # print("After training edge training loss: {}".format(np.sum(all_losses) / len(train_val_dataloader.dataset)))
    return np.sum(all_losses)

for i in range(20):
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    print("Train full model")
    for core in get_layer(model).construct_network():
        print(core.name, core.shape)
    train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler=None, n_epochs=args.full_train, device=device)
    choose_and_increase_edge(model, "classifier", train_edge, args.slice_gain, device, get_layer=get_layer, set_layer=set_layer)
    print("After step {}".format(i))
    ranks = []
    for factor in get_layer(model).factors:
        ranks.append(factor.shape)
    print("--ranks '{}'".format(json.dumps(ranks)))
            

print("Tensor shapes")
for core in get_layer(model).construct_network():
    print(core.name, core.shape)

# print("Fine tuning")
# for param in model.parameters():
#     param.requires_grad = True
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=2e-4)
# def learning_rate(epoch):
#     if epoch <= 80:
#         return 1
#     elif epoch <= 130:
#         return 0.1
#     else:
#         return 0.01
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate, verbose=True)
# accuracies = train(model, train_dataloader, val_dataloader, criterion, optimizer, device, 180, scheduler, plot=False)

# print("Model is trained")

all_losses, predicted_labels, true_labels = predict(model, val_dataloader, criterion, device)
accuracy = accuracy_score(true_labels.to("cpu"), predicted_labels.to("cpu"))
print(f"Accuracy: {accuracy}")
# plt.plot(accuracies)
# plt.show()

# torch.save({
#     "model": model.state_dict(),
# }, "./models/resnet-gap.1.pt")