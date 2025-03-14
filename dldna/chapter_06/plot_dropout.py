# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split


# def generate_data(seed=42):
#     """Generates synthetic data for the experiment."""
#     np.random.seed(seed)
#     X = np.random.randn(1000, 20).astype(np.float32)
#     y = np.random.randint(0, 2, 1000).astype(np.float32)
#     return train_test_split(X, y, test_size=0.2, random_state=seed)


# def prepare_data_loaders(X_train, X_test, y_train, y_test):
#     """Converts data to PyTorch tensors and creates data loaders."""
#     X_train_tensor = torch.FloatTensor(X_train)
#     y_train_tensor = torch.FloatTensor(y_train)
#     X_test_tensor = torch.FloatTensor(X_test)
#     y_test_tensor = torch.FloatTensor(y_test)

#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     return train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


# class ModelWithoutDropout(nn.Module):
#     """Defines a model without dropout."""
#     def __init__(self):
#         super(ModelWithoutDropout, self).__init__()
#         self.fc1 = nn.Linear(20, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return torch.sigmoid(self.fc3(x))


# class ModelWithDropout(nn.Module):
#     """Defines a model with dropout."""
#     def __init__(self, dropout_rate=0.5):
#         super(ModelWithDropout, self).__init__()
#         self.fc1 = nn.Linear(20, 64)
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.fc2 = nn.Linear(64, 32)
#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.fc3 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout2(x)
#         return torch.sigmoid(self.fc3(x))


# def train_model(model, optimizer, criterion, train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, epochs=100):
#     """Trains the model and returns training statistics."""
#     train_losses = []
#     train_accuracies = []
#     val_accuracies = []

#     for _ in range(epochs):
#         model.train()
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X).squeeze()
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()

#         model.eval()
#         with torch.no_grad():
#             train_outputs = model(X_train_tensor).squeeze()
#             train_loss = criterion(train_outputs, y_train_tensor)
#             train_acc = ((train_outputs > 0.5) == y_train_tensor).float().mean()

#             val_outputs = model(X_test_tensor).squeeze()
#             val_acc = ((val_outputs > 0.5) == y_test_tensor).float().mean()

#         train_losses.append(train_loss.item())
#         train_accuracies.append(train_acc.item())
#         val_accuracies.append(val_acc.item())

#     return train_losses, train_accuracies, val_accuracies


# def create_results_dataframe(train_acc, val_acc, model_name):
#     """Creates a Pandas DataFrame to store the results."""
#     return pd.DataFrame({
#         'epoch': range(1, 101),
#         'accuracy': train_acc,
#         'val_accuracy': val_acc,
#         'model': model_name
#     })


# def plot_results(df):
#     """Plots the training and validation accuracy using Seaborn."""
#     colors = ['#FF6666', '#FF9999', '#6666FF', '#9999FF']
#     sns.set_style("whitegrid")
#     plt.figure(figsize=(7, 4))

#     for i, model_name in enumerate(['Without Dropout', 'With Dropout']):
#         subset = df[df.model == model_name]
#         sns.lineplot(data=subset, x='epoch', y='accuracy',
#                      color=colors[i*2], label=f'{model_name} (Train)', linewidth=4, marker='o', markersize=4)
#         sns.lineplot(data=subset, x='epoch', y='val_accuracy',
#                      color=colors[i*2 + 1], label=f'{model_name} (Val)', linewidth=4, linestyle='--', marker='s', markersize=4)

#     plt.title('Accuracy(Train, Val): With vs Without Dropout', fontsize=14)
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Accuracy', fontsize=12)
#     plt.legend(title='Model & Metric', loc='lower right', fontsize=10, title_fontsize=12)
#     plt.tick_params(axis='both', which='major', labelsize=10)
#     plt.ylim(0.4, 1.0)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xticks(range(0, 101, 10))
#     plt.tight_layout()
#     plt.show()


# def plot_dropout_effect():
#     """Main function to demonstrate the effect of dropout."""
#     X_train, X_test, y_train, y_test = generate_data()
#     train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data_loaders(X_train, X_test, y_train, y_test)

#     model_without_dropout = ModelWithoutDropout()
#     model_with_dropout = ModelWithDropout()

#     criterion = nn.BCELoss()
#     optimizer_without_dropout = optim.Adam(model_without_dropout.parameters())
#     optimizer_with_dropout = optim.Adam(model_with_dropout.parameters())

#     _, train_acc_without, val_acc_without = train_model(
#         model_without_dropout, optimizer_without_dropout, criterion, train_loader,
#         X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
#     )
#     _, train_acc_with, val_acc_with = train_model(
#         model_with_dropout, optimizer_with_dropout, criterion, train_loader,
#         X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
#     )

#     df_without_dropout = create_results_dataframe(train_acc_without, val_acc_without, 'Without Dropout')
#     df_with_dropout = create_results_dataframe(train_acc_with, val_acc_with, 'With Dropout')
#     df = pd.concat([df_without_dropout, df_with_dropout])

#     plot_results(df)



# if __name__ == '__main__':
#     plot_dropout_effect()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_data(seed=42, n_samples=2000, n_features=30):
    """Generates synthetic data with more features and samples, making
    overfitting more likely without dropout.
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Introduce some non-linearity and feature interactions to make it harder
    y = (np.sin(X[:, 0] * 2) + X[:, 1] * X[:, 2] + np.random.randn(n_samples) * 0.2 > 0).astype(np.float32)

    return train_test_split(X, y, test_size=0.3, random_state=seed)  # Larger test set


def prepare_data_loaders(X_train, X_test, y_train, y_test, batch_size=64): # Larger batch size
    """Converts data to PyTorch tensors, scales the features, and
    creates data loaders.
    """
    # Feature Scaling (Important for better convergence)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor) # Test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Test loader
    return train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


class ModelWithoutDropout(nn.Module):
    """Defines a deeper model without dropout (more layers, more units)."""
    def __init__(self, n_features):
        super(ModelWithoutDropout, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 256)  # More units
        self.fc3 = nn.Linear(256, 128)  # Additional layer
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


class ModelWithDropout(nn.Module):
    """Defines a model with dropout, and adjustable dropout rate."""
    def __init__(self, n_features, dropout_rate=0.5):
        super(ModelWithDropout, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, 1)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        return torch.sigmoid(self.fc4(x))

def train_model(model, optimizer, criterion, train_loader, X_train_tensor, y_train_tensor,
                X_test_tensor, y_test_tensor, epochs=200, patience=20):  # Early Stopping
    """Trains the model with early stopping."""
    train_losses = []
    train_accuracies = []
    val_losses = []  # Validation loss
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0  # For calculating average loss per epoch
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)  # Accumulate loss
            predicted = (outputs > 0.5).float()
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)



        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor).squeeze()
            val_loss = criterion(val_outputs, y_test_tensor)
            val_acc = ((val_outputs > 0.5) == y_test_tensor).float().mean()

            val_losses.append(val_loss.item())
            val_accuracies.append(val_acc.item())

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict() # Save best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    model.load_state_dict(best_model_state) # Load best model
    return train_losses, train_accuracies, val_losses, val_accuracies, epoch + 1


def create_results_dataframe(train_acc, val_acc, model_name, epochs):
    """Creates a Pandas DataFrame, handling different epoch lengths."""
    # Use the actual number of epochs completed
    return pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'accuracy': train_acc,
        'val_accuracy': val_acc,
        'model': model_name
    })


def plot_results(df):
    """Plots the training and validation accuracy with improved aesthetics."""

    sns.set_theme(style="whitegrid", palette="muted")  # Use set_theme
    plt.figure(figsize=(10, 6))  # Larger figure

    for model_name in df['model'].unique():
        subset = df[df.model == model_name]
        # Plot training accuracy with markers
        sns.lineplot(data=subset, x='epoch', y='accuracy',
                     label=f'{model_name} (Train)', linewidth=2.5,
                     marker='o', markersize=6)
        # Plot validation accuracy with markers
        sns.lineplot(data=subset, x='epoch', y='val_accuracy',
                     label=f'{model_name} (Val)', linewidth=2.5,
                     linestyle='--', marker='s', markersize=6)


    plt.title('Training and Validation Accuracy: With vs Without Dropout', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.tick_params(labelsize=12)
    plt.ylim(0.4, 1.05)  # Adjust y-axis limits
    plt.xlim(0, df['epoch'].max())      # Set x-axis limits to max epoch
    plt.grid(True, which="both", ls="--", c='0.7') # Lighter grid
    plt.tight_layout()
    plt.show()


def plot_dropout_effect():
    """Main function demonstrating dropout with improved visualization."""
    X_train, X_test, y_train, y_test = generate_data()
    train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data_loaders(X_train, X_test, y_train, y_test)

    n_features = X_train.shape[1]
    model_without_dropout = ModelWithoutDropout(n_features)
    model_with_dropout = ModelWithDropout(n_features, dropout_rate=0.5)  # Adjustable rate

    criterion = nn.BCELoss()
    optimizer_without_dropout = optim.Adam(model_without_dropout.parameters(), lr=0.001) # learning rate
    optimizer_with_dropout = optim.Adam(model_with_dropout.parameters(), lr=0.001)

    train_losses_without, train_acc_without, val_losses_without, val_acc_without, epochs_without = train_model(
        model_without_dropout, optimizer_without_dropout, criterion, train_loader,
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    )
    train_losses_with, train_acc_with, val_losses_with, val_acc_with, epochs_with = train_model(
        model_with_dropout, optimizer_with_dropout, criterion, train_loader,
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    )

    df_without_dropout = create_results_dataframe(train_acc_without, val_acc_without, 'Without Dropout', epochs_without)
    df_with_dropout = create_results_dataframe(train_acc_with, val_acc_with, 'With Dropout', epochs_with)
    df = pd.concat([df_without_dropout, df_with_dropout])

    plot_results(df)




if __name__ == '__main__':
    plot_dropout_effect()