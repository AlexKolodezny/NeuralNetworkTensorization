import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import ssl
import argparse
from src.utils import create_dataset, set_random_seed, train, predict
import tensornetwork as tn
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", default=12345, type=int)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--filename", help="File to store checkpoints", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--full_train", default=2, type=int)
parser.add_argument("--edge_train", default=2, type=int)
args = parser.parse_args()


print("Initializing dataloader")
ssl._create_default_https_context = ssl._create_unverified_context
train_dataset, val_dataset = create_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
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

from src.greedy_tn import increase_edge_in_layer

def choose_and_increase_edge(model, tensor_network, train_edge):
    edges = list(tn.get_all_nondangling(tensor_network))
    edges_results = []
    old_layer = model.classifier
    for edge in edges:
        model.classifier, cloned_edge = increase_edge_in_layer(old_layer, edge, device=device)

        for param in model.parameters():
            param.requires_grad = False
        cloned_edge.node1.tensor.requires_grad = True
        cloned_edge.node2.tensor.requires_grad = True

        print("Train edge {}".format(edge.name))
        final_loss = train_edge(model)
        model.classifier.clear_masks()
        edges_results.append((model.classifier, final_loss, edge.name))
    model.classifier, _, edgename = min(edges_results, key=lambda x: x[1])
    print("Choosen edge {}".format(edgename))

print("Warm up model")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=2e-4)
def learning_rate(epoch):
    if epoch < 10:
        return epoch * 0.1 + 0.1
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
    n_epochs=10,
    start_epoch=0,
    device=device)

print("Stop warm up model")

print("Choosign ranks")
def train_edge(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler=None, n_epochs=args.edge_train, device=device)
    all_losses, _, _ = predict(model, train_dataloader, criterion, device)
    print("After training edge training loss: {}".format(np.sum(all_losses) / len(train_dataloader.dataset)))
    return np.sum(all_losses)

for i in range(20):
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    print("Train full model")
    print("Before training full model training loss: {}".format(np.sum(all_losses) / len(train_dataloader.dataset)))
    train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler=None, n_epochs=args.full_train, device=device)
    all_losses, _, _ = predict(model, train_dataloader, criterion, device)
    print("After training full model training loss: {}".format(np.sum(all_losses) / len(train_dataloader.dataset)))
    choose_and_increase_edge(model, model.classifier.construct_network(), train_edge)

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