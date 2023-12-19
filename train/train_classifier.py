from data.datasets import LakhDataset
from models.models import Classifier
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import tqdm
import os
import matplotlib.pyplot as plt

# hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_accuracy = 0

    # iterate over the training dataloader
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        optimizer.zero_grad()
        input_tokens, labels = batch
        input_tokens = input_tokens.to(device).long()
        labels = labels.to(device).long()

        # forward pass
        logits = model(input_tokens)

        # calculate the loss
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # calculate the accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(predictions == labels) / len(labels)

        total_loss += loss.item()
        total_accuracy += accuracy.item()

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)

    return average_loss, average_accuracy

def validate_loop(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    # iterate over the training dataloader
    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        input_tokens, labels = batch
        input_tokens = input_tokens.to(device).long()
        labels = labels.to(device).long()

        # forward pass
        logits = model(input_tokens)

        # calculate the loss
        loss = loss_fn(logits, labels)

        # calculate the accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(predictions == labels) / len(labels)

        total_loss += loss.item()
        total_accuracy += accuracy.item()

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)

    return average_loss, average_accuracy



def plot_curves(train, val, title):
    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Training')
    plt.plot(val, label='Validation')
    plt.title(title)
    plt.legend()
    plt.show()
    return plt

def train(num_labels=13, num_epochs=20):
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Load the custom model and move it to the specified device (GPU if available)
    model = Classifier('stanford-crfm/music-small-800k', num_labels=num_labels).to(device)

    # create the dataset
    dataset = LakhDataset(
        root_dir='data/lmd_tokenized',
        label_path='data/genre_labels.csv',
        sequence_length=1024
    )

    # split the dataset into training and validation sets
    train_set, val_set = random_split(dataset, [0.8, 0.2])

    # create the training and validation dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # create the loss function
    loss = torch.nn.CrossEntropyLoss()

    # lists to keep track of training and validation losses and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # train the model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        train_loss, train_accuracy = train_loop(train_loader, model, loss, optimizer, device)
        val_loss, val_accuary = validate_loop(val_loader, model, loss, device)

        # append the losses and accuracies to the lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuary)

    # save the model
    output_dir = 'models/weights'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f'classifier-{num_epochs}-epochs.pt'))

    # plot the training and validation losses and accuracies
    loss_curve = plot_curves(train_losses, val_losses, 'Losses')
    accuracy_curve = plot_curves(train_accuracies, val_accuracies, 'Accuracies')

    # save the plots
    output_dir = 'models/plots'
    os.makedirs(output_dir, exist_ok=True)
    loss_curve.savefig(os.path.join(output_dir, f'classifier-{num_epochs}-epochs-loss.png'))
    accuracy_curve.savefig(os.path.join(output_dir, f'classifier-{num_epochs}-epochs-accuracy.png'))

if __name__ == '__main__':
    train()