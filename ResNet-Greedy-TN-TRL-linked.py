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
from src.layers.trl_linked import TRLMasked


print("Initializing model")
set_random_seed(args.seed)
model = resnet32()
model.classifier = TRLMasked((64, 8, 8), (1, 1, 1), (1, 1, 1), 10)
start_epoch = 0

criterion = nn.CrossEntropyLoss(reduction="mean")

device = args.device if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print("Model initialized")

for core in model.classifier.construct_network():
    print(core.name, core.shape)

print("Testing model")
all_losses, predicted_labels, true_labels = predict(model, val_dataloader, criterion, device)
assert len(predicted_labels) == len(val_dataset)
accuracy = accuracy_score(predicted_labels.to("cpu"), true_labels.to("cpu"))
print("Tests passed")

from src.greedy_tn import choose_and_increase_edge

# def choose_and_increase_edge(model, tensor_network, train_edge):
#     edges = list(tn.get_all_nondangling(tensor_network))
#     edges_results = []
#     # old_layer = deepcopy(model.classifier)
#     copy_model = deepcopy(model)
#     for edge in edges:
#         model = deepcopy(copy_model)

#         for param in model.parameters():
#             param.requires_grad = False
#         cloned_edge.node1.tensor.requires_grad = True
#         cloned_edge.node2.tensor.requires_grad = True

#         print("Train edge {}".format(edge.name))
#         final_loss = train_edge(model)
#         model.classifier.clear_masks()
#         # result_cores = {node.name: node for node in model.classifier.construct_network()}
#         # prev_cores = {node.name: node for node in copy_model.classifier.construct_network()}
#         # for name in result_cores.keys():
#         #     if edge.node1.name != name and edge.node2.name != name:
#         #         assert torch.allclose(result_cores[name].tensor, prev_cores[name].tensor)
#         #     else:
#         #         assert torch.allclose(result_cores[name].tensor[(0,) * len(result_cores[name].shape)], prev_cores[name].tensor[(0,) * len(prev_cores[name].shape)])
#         #         assert result_cores[name].shape != prev_cores[name].shape
#         edges_results.append((model.classifier, final_loss, edge.node1.name, edge.node2.name))
#     model.classifier, _, edgenode1, edgenode2 = min(edges_results, key=lambda x: x[1])
#     # result_cores = {node.name: node for node in model.classifier.construct_network()}
#     # prev_cores = {node.name: node for node in copy_model.classifier.construct_network()}
#     # for name in result_cores.keys():
#     #     if edgenode1 != name and edgenode2 != name:
#     #         assert torch.allclose(result_cores[name].tensor, prev_cores[name].tensor)
#     #     else:
#     #         assert torch.allclose(result_cores[name].tensor[(0,) * len(result_cores[name].shape)], prev_cores[name].tensor[(0,) * len(prev_cores[name].shape)])
#     #         assert result_cores[name].shape != prev_cores[name].shape
#     print("Choosen edge between {} and {}".format(edgenode1, edgenode2))
#     for param in model.parameters():
#         param.requires_grad = True

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
    for core in model.classifier.construct_network():
        print(core.name, core.shape)
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    all_losses, _, _ = predict(model, train_val_dataloader, criterion, device)
    print("Before train edge training loss: {}".format(np.sum(all_losses) / len(train_val_dataloader.dataset)))
    train(model, train_val_dataloader, val_dataloader, criterion, optimizer, scheduler=None, n_epochs=args.edge_train, device=device)
    all_losses, _, _ = predict(model, train_val_dataloader, criterion, device)
    print("After training edge training loss: {}".format(np.sum(all_losses) / len(train_val_dataloader.dataset)))
    return np.sum(all_losses)

for i in range(20):
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    print("Train full model")
    all_losses, _, _ = predict(model, train_val_dataloader, criterion, device)
    print("Before training full model training loss: {}".format(np.sum(all_losses) / len(train_val_dataloader.dataset)))
    for core in model.classifier.construct_network():
        print(core.name, core.shape)
    train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler=None, n_epochs=args.full_train, device=device)
    all_losses, _, _ = predict(model, train_val_dataloader, criterion, device)
    print("After training full model training loss: {}".format(np.sum(all_losses) / len(train_val_dataloader.dataset)))
    choose_and_increase_edge(model, "classifier", train_edge, args.slice_gain, device)
    ring_ranks = []
    for core in model.classifier.construct_network():
        if core.name == "core":
            core_ranks = (core["rank_0"].dimension, core["rank_1"].dimension, core["rank_2"].dimension)
        else:
            ring_ranks.append(core["link_right"].dimension)
    ring_ranks = tuple(map(str, ring_ranks))
    core_ranks = tuple(map(str, core_ranks))
    print("After step {}".format(i))
    print("--core_ranks", " ".join(core_ranks), "--ring_ranks", " ".join(ring_ranks))
            

print("Tensor shapes")
for core in model.classifier.construct_network():
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