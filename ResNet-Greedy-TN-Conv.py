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
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--full_train", default=2, type=int)
parser.add_argument("--edge_train", default=2, type=int)
parser.add_argument("--slice_gain", default=1, type=float)
parser.add_argument("--warmup", default=10, type=int)
parser.add_argument("--learning_rate", default=0.1, type=float)
parser.add_argument("--iterations", default=20, type=int)
# parser.add_argument("--tensorization", nargs="+", type=int, default=[8, 8, 8, 8])
# parser.add_argument("--pretrained_filename", type=str, default=None)
parser.add_argument("--convolution", type=str, default="tt")
# parser.add_argument("--network", type=str, default="resnet")
args = parser.parse_args()

print(sys.argv)

print("Initializing dataloader")
ssl._create_default_https_context = ssl._create_unverified_context
train_dataset, val_dataset = create_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
train_val_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=128)
print("Dataloader has initialized")

from src.networks.resnet_tensorized import tensorized_resnet32
from src.networks.wide_resnet import Wide_ResNet
from src.layers.full_linked import FullLinkedLayer

# n_factors = len(args.tensorization) + 1

print("Initializing model")
set_random_seed(args.seed)
model = tensorized_resnet32((1, 1), 1)
# if args.network == "resnet":
#     model = tensorized_resnet32()
# else: 
#     assert False, "Unknown network"
# if args.pretrained_filename is not None:
#     if args.network == "resnet":
#         model.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(4096, 10),
#         )
#     elif args.network == "wide_resnet":
#         model.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(4096 * 8, 10),
#         )
#     path = "./models/" + args.pretrained_filename + ".pt"
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint["model"])

# set_random_seed(args.seed)
# if args.classifier == "full":
#     model.classifier = nn.Sequential(
#         nn.Flatten(),
#         nn.Unflatten(1, tuple(args.tensorization)),
#         FullLinkedLayer(tuple(args.tensorization), (10,), [[1] * n_factors for i in range(n_factors)]),
#     )
# elif args.classifier == "ringed":
#     model.classifier = nn.Sequential(
#         nn.Flatten(),
#         nn.Unflatten(1, tuple(args.tensorization)),
#         TRLRinged(tuple(args.tensorization), (1,) * len(args.tensorization), (1,) * len(args.tensorization), 10),
#     )
# else:
#     print("Unknown classifier")
#     assert False

class GetLayer:
    def __init__(self, layer_num, block_num, conv_num):
        self.layer_num = layer_num
        self.block_num = block_num
        self.conv_num = conv_num
    
    def __call__(self, model):
        return getattr(getattr(model, f"layer{self.layer_num}")[self.block_num], f"conv{self.conv_num}")

class SetLayer:
    def __init__(self, layer_num, block_num, conv_num):
        self.layer_num = layer_num
        self.block_num = block_num
        self.conv_num = conv_num
    
    def __call__(self, model, new_layer):
        setattr(getattr(model, f"layer{self.layer_num}")[self.block_num], f"conv{self.conv_num}", new_layer)

get_set_layers = [
    (GetLayer(layer_num, block_num, conv_num),
    SetLayer(layer_num, block_num, conv_num))
    for layer_num in range(1, 4)
    for conv_num in range(1, 3)
    for block_num in range(len(model.layer1))]

start_epoch = 0

criterion = nn.CrossEntropyLoss(reduction="mean")

device = args.device if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print("Model initialized")

for core in get_set_layers[0][0](model).construct_network():
    print(core.name, core.shape)

print("Testing model")
all_losses, predicted_labels, true_labels = predict(model, val_dataloader, criterion, device)
assert len(predicted_labels) == len(val_dataset)
accuracy = accuracy_score(predicted_labels.to("cpu"), true_labels.to("cpu"))
print("Tests passed")

from src.greedy_tn import choose_and_increase_edge_on_layers

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
# if args.pretrained_filename is not None:
#     for param in model.parameters():
#         param.requires_grad = False
#     for param in get_layer(model).parameters():
#         param.requires_grad = True
# else:
#     for param in model.parameters():
#         param.requires_grad = True
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
    for core in get_set_layers[0][0](model).construct_network():
        print(core.name, core.shape)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    print("Before train edge")
    train(model, train_val_dataloader, val_dataloader, criterion, optimizer, scheduler=None, n_epochs=args.edge_train, device=device)
    print("After training edge")
    return np.sum(all_losses)

for i in range(args.iterations):
    # if args.pretrained_filename is not None:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for param in get_layer(model).parameters():
    #         param.requires_grad = True
    # else:
    #     for param in model.parameters():
    #         param.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    print("Train full model")
    for core in get_set_layers[0][0](model).construct_network():
        print(core.name, core.shape)
    train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler=None, n_epochs=args.full_train, device=device)
    choose_and_increase_edge_on_layers(model, train_edge, args.slice_gain, device, get_set_layers)
    print("After step {}".format(i))
    ranks = []
    for factor in get_set_layers[0][0](model).factors:
        ranks.append(factor.shape[2])
    space_rank = get_set_layers[0][0](model).space_factor.shape[0]
    print("--ranks {} --space_rank {}".format(" ".join(map(str, ranks)), space_rank))

print("Tensor shapes")
for core in get_set_layers[0][0](model).construct_network():
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