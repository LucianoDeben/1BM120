import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

def train_model(model, dataset, criterion, optimizer, device, n_epochs=10, val_split=0.1, model_name="simple_model"):
    """
    Train a PyTorch model on a dataset
    
    Args:
    - model: PyTorch model
    - dataset: PyTorch dataset
    - criterion: loss function
    - optimizer: PyTorch optimizer
    - device: 'cpu' or 'cuda'
    - n_epochs: number of epochs to train the model
    - val_split: fraction of dataset to use as validation set
    - model_name: name to save the model

    Returns:
    - train_losses: list of training losses
    - val_losses: list of validation losses
    """
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=False)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{n_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        avg_val_loss = running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"models/{model_name}.pth")

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    print('Finished Training')
    return train_losses, val_losses


def test_model(model, test_data, device):
    """
    Test a PyTorch model on a dataset
    
    Args:
    - model: PyTorch model
    - test_data: PyTorch dataset
    - device: 'cpu' or 'cuda'
    
    Returns:
    - accuracy: accuracy of the model on the test set
    - precision: precision of the model on the test set
    - recall: recall of the model on the test set
    - f1: F1 score of the model on the test set
    """
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, drop_last=False)
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1

def objective_with_cv(trial, model_class, train_data, n_epochs=10, device='cpu'):
    """
    Objective function for Optuna to optimize hyperparameters
    
    Args:
    - trial: Optuna trial
    - model: PyTorch model class
    - train_data: PyTorch dataset
    - n_epochs: number of epochs to train the model
    - device: 'cpu' or 'cuda'
    
    Returns:
    - mean_accuracy: mean accuracy of the model on a 5-fold cross-validation
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=5)
    accuracies = []

    for train_index, val_index in kf.split(train_data):
        train_subset = Subset(train_data, train_index)
        val_subset = Subset(train_data, val_index)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize model, criterion, and optimizer
        model = model_class(dropout_rate=dropout_rate).to(device)
        criterion = nn.BCELoss()
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Training loop
        for epoch in range(n_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)  # Calculate mean accuracy
    return mean_accuracy