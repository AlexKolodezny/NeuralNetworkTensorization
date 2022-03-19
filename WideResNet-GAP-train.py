import defusedxml
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import ssl
import argparse
from src.utils import create_dataset, set_random_seed, train, predict


parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", default=12345, type=int)
parser.add_argument("--depth", help="Wide ResNet Depth", default=16, type=int)
parser.add_argument("--widening_factor", default=1, type=int)
parser.add_argument("--n_epochs", help="number of epochs", default=200, type=int)
parser.add_argument("--dropout", default=0.3, type=float)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--filename", help="File to store checkpoints", type=str)
args = parser.parse_args()


print("Initializing dataloader")
ssl._create_default_https_context = ssl._create_unverified_context
train_dataset, val_dataset = create_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128)
print("Dataloader has initialized")

from src.networks.wide_resnet import Wide_ResNet

print("Initializing model")
if args.resume:
    model = Wide_ResNet(args.depth, args.widening_factor, args.dropout, 10)
    assert args.filename is not None, "Filename required for resume"
    path = "./models/" + args.filename + ".pt"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    start_epoch = checkpoint["epoch"] + 1
else:
    set_random_seed(args.seed)
    model = Wide_ResNet(args.depth, args.widening_factor, args.dropout, 10)
    start_epoch = 0

optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4, nesterov=True)
criterion = nn.CrossEntropyLoss(reduction="mean")
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
n_epochs = args.n_epochs

device = "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
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