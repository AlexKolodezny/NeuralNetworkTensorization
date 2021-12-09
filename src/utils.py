from tqdm.notebook import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random


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
          losses.append(criterion(logit, label) * len(images))
          preds.append(logit.argmax(dim=1))
          labels.append(label)
    return losses, torch.cat(preds), torch.cat(labels)


def train(model, train_dataloader, val_dataloader, criterion, optimizer, device="cuda:0", n_epochs=10, scheduler=None, plot=False):
    accuracies = []
    for epoch in range(n_epochs):
        train_losses, sizes = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        if plot:
            plt.plot(np.array(train_losses) / np.array(sizes))
            plt.show()
        val_losses, val_pred, true_pred = predict(model, val_dataloader, criterion, device)
        val_accuracy = accuracy_score(true_pred.to("cpu"), val_pred.to("cpu"))
        accuracies.append(val_accuracy)
        if scheduler is not None:
            scheduler.step()
        print("Epoch: {}, Train loss: {}, Validation loss: {}, Validation accuracy: {}"\
              .format(epoch, sum(train_losses) / len(train_dataloader.dataset), sum(val_losses) / len(val_dataloader.dataset), val_accuracy))
    return accuracies