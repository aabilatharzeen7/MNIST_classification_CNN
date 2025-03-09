"""
Module for training a CNN on the MNIST dataset
This module provides helper functions

"""
import torch
import torchvision
from torchvision import transforms
import numpy as np
from torch import  nn, optim
import torch.nn.functional as f
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")




def evaluate(model, device, val_loader):
    """
    Function to evaluate the test data

    """


    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += f.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    len_val_loader = sum(data.size(0) for data, target in val_loader)
    val_loss /= len_val_loader

    return correct / len_val_loader, val_loss
def get_data (val_split):
    """
    Load and normalizes the data and also creates training, validation
    and test data set.

    """
    # Define a transform to normalize the data
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])  # global mean = 0.1307 and global stdev = 0.3081
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=True, transform=transform)

    # Split the training dataset into training and validation sets
    num_train = len(train_dataset)
    indices = list(range(num_train))

    # Create subsets for training and validation
    np.random.shuffle(indices)
    split = int(np.floor(val_split * num_train))
    val_indices = indices[:split]
    train_indices = indices[split:]
    train_data = [train_dataset[i] for i in train_indices]
    val_dataset = [train_dataset[i] for i in val_indices]


    return train_data, val_dataset, test_dataset


def create_batches(dataset, batch_size, shuffle=True, device='cpu'):
    """
    Generator that yields mini-batches from the dataset.
    Each batch is a tuple (images, labels).
    """
    if shuffle:
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
    else:
        indices = np.arange(len(dataset))

    for start_idx in range(0, len(dataset), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch_images = []
        batch_labels = []
        for idx in batch_indices:
            image, label = dataset[idx]
            batch_images.append(image.to(device))
            batch_labels.append(label)
        # Stack list of tensors into one tensor of shape [batch_size, channels, height, width]
        images = torch.stack(batch_images)
        labels = torch.tensor(batch_labels, device=device)
        yield images, labels


class CnnModel( nn.Module):
    """
    CNN based Model for MNIST classification

    """
    def __init__(self,args):
        super(CnnModel, self).__init__()
        self.conv1 = nn.Conv2d(1, args.hidden_1, 3, 1)
        self.conv2 = nn.Conv2d(args.hidden_1, args.hidden_2, 3, 1)
        self.dropout1 = nn.Dropout(args.dropout1)
        self.dropout2 = nn.Dropout(args.dropout2)
        # the output after two convolution layers and a 2d maxpool ((w-k+2p)/s)+1 = 12
        self.fc1 = nn.Linear(args.hidden_2*12*12, args.hidden_3)
        self.fc2 = nn.Linear(args.hidden_3, 10)

    def forward(self, x):
        """ Forward pass of the architecture """
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output



def run(args,device):
    """
    Loads data and train the model

    """

    train_data, val_data, test_data = get_data(args.val_split)

    # Create batches for training, validation, and testing.
    train_loader = list(create_batches(train_data, args.batch_size, shuffle=True, device=device))
    val_loader = list(create_batches(val_data, args.batch_size, shuffle=False, device=device))
    test_loader = list(create_batches(test_data,args.test_batch_size, shuffle=False, device=device))

    model = CnnModel(args).to(device)

    # Early stopping parameters
    patience = 10
    best_loss = float('inf')
    epochs_no_improve = 0

    # Track average loss during each epoch
    avg_train_losses = []
    avg_valid_losses = []

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs):

        # Track the batch losses
        train_losses = []
        valid_losses = []

        train_correct = 0   # calculate train accuracy

        # Train the model
        model.train()
        for _, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            loss = f.nll_loss(output, target)   # nll loss used since log-softmax used in the model
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())


        # Validate the model
        model.eval()
        val_correct = 0 # calculate validation accuracy
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_losses.append(f.nll_loss(output, target).item())  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get index of the max log-probability
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        len_val_loader = sum(data.size(0) for data, target in val_loader)
        len_train_loader = sum(data.size(0) for data, target in train_loader)


        # calculate average loss over an epoch
        avg_train_losses.append(np.average(train_losses))
        avg_valid_losses.append(np.average(valid_losses))

        # Early Stopping Check
        if np.average(valid_losses) < best_loss:
            best_loss = np.average(valid_losses)
            epochs_no_improve = 0


        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break


        print_msg = (f'[Epoch: {epoch}] ' +
                     f'train_loss: {np.average(train_losses):.5f} ' +
                     f'valid_loss: {np.average(valid_losses):.5f}')

        print(print_msg)

    # Optionally, save the best model state
    if args.save_model:
        torch.save(model.state_dict(), "mnist_best_model.pt")

    # Optionally, If losses to be plotted
    if args.plot_loss:
        plt.figure(figsize=(12, 6))
        plt.plot(avg_train_losses, label='Training')
        plt.plot(avg_valid_losses, label='Validation')
        plt.xlabel('Num of epochs')
        plt.ylabel('Losses')
        plt.title('Training and validation loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()

    if args.save_model:
        model.load_state_dict(torch.load("mnist_best_model.pt"))

    test_accuracy, test_loss = evaluate(model, device, test_loader)

    return (np.average(train_losses),
            np.average(valid_losses),
            test_loss,
            100. *train_correct/len_train_loader,
            100. *val_correct/len_val_loader,
            100. *test_accuracy)
