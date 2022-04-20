import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import ssl
import argparse
from src.utils import create_dataset, set_random_seed, train, predict
import sys
import json


parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", default=12345, type=int)
parser.add_argument("--n_epochs", help="number of epochs", default=200, type=int)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--filename", help="File to store checkpoints", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--ranks", default="[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]", type=str)
parser.add_argument("--milestones", default=[100, 150], type=int, nargs="+")
args = parser.parse_args()

print(sys.argv)

print("Initializing dataloader")
ssl._create_default_https_context = ssl._create_unverified_context
train_dataset, val_dataset = create_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128)
print("Dataloader has initialized")

from src.networks.resnet_tensorized_full import tensorized_resnet32

print("Initializing model")
if args.resume:
    model = tensorized_resnet32(json.loads(args.ranks))
    assert args.filename is not None, "Filename required for resume"
    path = "./models/" + args.filename + ".pt"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    start_epoch = checkpoint["epoch"] + 1
else:
    set_random_seed(args.seed)
    model = tensorized_resnet32(json.loads(args.ranks))
    start_epoch = 0

optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=2e-4)
criterion = nn.CrossEntropyLoss(reduction="mean")
def learning_rate(epoch):
    if epoch < 10:
        return epoch * 0.1 + 0.1
    lr = 1
    for milestone in args.milestones:
        if epoch <= milestone:
            return lr
        lr *= 0.1
    return lr
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate, verbose=True, last_epoch=start_epoch-1)
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
print(f"Accuracy: {accuracy}")
# plt.plot(accuracies)
# plt.show()

# torch.save({
#     "model": model.state_dict(),
# }, "./models/resnet-gap.1.pt")