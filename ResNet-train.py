import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import ssl
import argparse
from src.utils import create_dataset, set_random_seed, train, predict
import sys
import json

def calculate_parameters(model):
    import numpy as np
    result = 0
    for param in model.parameters():
        result += np.prod(param.shape)
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--classifier", help="type of tensor network", type=str)
parser.add_argument("--seed", help="random seed", default=12345, type=int)
parser.add_argument("--n_epochs", help="number of epochs", default=200, type=int)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--filename", help="File to store checkpoints", type=str)
parser.add_argument("--ranks", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--tensorization", default=[64, 8, 8], nargs="+", type=int)
parser.add_argument("--core_ranks", nargs="+", type=int)
parser.add_argument("--ring_ranks", nargs="+", type=int)
parser.add_argument("--network", type=str, default="resnet")
args = parser.parse_args()

print(sys.argv)

print("Initializing dataloader")
ssl._create_default_https_context = ssl._create_unverified_context
train_dataset, val_dataset = create_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128)
print("Dataloader has initialized")

from src.networks.resnet import resnet32
from src.networks.wide_resnet import Wide_ResNet
from src.layers.full_linked import FullLinkedLayer
from src.layers.trl_ringed import TRLRinged

print("Initializing model")
set_random_seed(args.seed)
if args.network == "resnet":
    model = resnet32()
elif args.network == "wide_resnet":
    model = Wide_ResNet(16, 8, 0.3, 10)
else: 
    assert False, "Unknown network"
if args.classifier == "gap":
    pass
elif args.classifier == "linear":
    if args.network == "resnet":
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 10),
        )
    elif args.network == "wide_resnet":
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096 * 8, 10),
        )
elif args.classifier == "full":
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Unflatten(1, tuple(args.tensorization)),
        FullLinkedLayer(tuple(args.tensorization), (10,), json.loads(args.ranks)),
    )
elif args.classifier == "ringed":
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Unflatten(1, tuple(args.tensorization)),
        TRLRinged(tuple(args.tensorization), tuple(args.core_ranks), tuple(args.ring_ranks), 10),
    )
else:
    print("Unknown classifier")
    assert False

if args.resume:
    assert args.filename is not None, "Filename required for resume"
    path = "./models/" + args.filename + ".pt"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    start_epoch = checkpoint["epoch"] + 1
else:
    start_epoch = 0

optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=2e-4)
criterion = nn.CrossEntropyLoss(reduction="mean")
def learning_rate_resnet(epoch):
    if epoch < 10:
        return epoch * 0.1 + 0.1
    if epoch <= 100:
        return 1
    elif epoch <= 150:
        return 0.1
    else:
        return 0.01
if args.network == "resnet":
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_resnet, verbose=True)
elif args.network == "wide_resnet":
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
n_epochs = args.n_epochs

device = args.device if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
print("Model initialized")

print("Testing model")
all_losses, predicted_labels, true_labels = predict(model, val_dataloader, criterion, device)
assert len(predicted_labels) == len(val_dataset)
accuracy = accuracy_score(predicted_labels.to("cpu"), true_labels.to("cpu"))
print("Tests passed")

accuracies = train(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    device,
    n_epochs,
    scheduler,
    plot=False,
    start_epoch=start_epoch,
    filename=args.filename)

all_losses, predicted_labels, true_labels = predict(model, val_dataloader, criterion, device)
accuracy = accuracy_score(true_labels.to("cpu"), predicted_labels.to("cpu"))
print("Accuracy: {}, Parameters: {}".format(accuracy, calculate_parameters(model.classifier)))
# plt.plot(accuracies)
# plt.show()

# torch.save({
#     "model": model.state_dict(),
# }, "./models/resnet-gap.1.pt")