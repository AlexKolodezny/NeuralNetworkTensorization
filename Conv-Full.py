import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import ssl
import argparse
from src.utils import create_dataset, set_random_seed, train, predict


parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", default=12345, type=int)
parser.add_argument("--tucker_channel_rank", help="Convolution Tucker rank", nargs="+", default=[1, 1], type=int)
parser.add_argument("--tucker_space_rank", type=int, default=1)
parser.add_argument("--tttf_space_rank", help="TTTF space rank", default=1, type=int)
parser.add_argument("--tttf_channel_rank", default=[1, 1], nargs="+", type=int)
parser.add_argument("--n_epochs", help="number of epochs", default=100, type=int)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--filename", help="File to store checkpoints", type=str)
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()


print("Initializing dataloader")
ssl._create_default_https_context = ssl._create_unverified_context
train_dataset, val_dataset = create_dataset()
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128)
print("Dataloader has initialized")

from src.layers.tttf import TTTF
from src.layers.tucker_conv import TuckerConv

args.tucker_channel_rank = tuple(args.tucker_channel_rank)

set_random_seed(args.seed)
model = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(3, 64, (3, 3), padding="same"),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Conv2d(64, 64, (3, 3), padding="same"),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    ),
    nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
    nn.Sequential(
        nn.Conv2d(64, 128, (3, 3), padding="same"),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Conv2d(128, 128, (3, 3), padding="same"),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    ),
    nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
    nn.Sequential(
        nn.Conv2d(128, 128, (3, 3), padding="same"),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Conv2d(128, 128, (3, 3), padding="same"),
        nn.BatchNorm2d(128),
        nn.ReLU(),
    ),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(8192, 10),
    )
)



print("Initializing model")
if args.resume:
    assert args.filename is not None, "Filename required for resume"
    path = "./models/" + args.filename + ".pt"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    start_epoch = checkpoint["epoch"] + 1
else:
    start_epoch = 0

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss(reduction="mean")
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1, verbose=True)
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