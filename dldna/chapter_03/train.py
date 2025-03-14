# dld/chapter_03/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from dldna.chapter_03.models import ConvNet  # Import the model
from dldna.chapter_03.visualization.tensorboard_utils import TensorboardLogger, VisualizationUtils, get_embeddings_and_labels  # Import logger
from sklearn.manifold import TSNE

def train(hparams_dict=None, log_dir=None):
    """
    Trains the ConvNet model on the MNIST dataset and logs the training
    process using TensorBoard.
    """
    if hparams_dict is None:  # Default hyperparameters
        hparams_dict = {
            'batch_size': 64,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'epochs': 10,  # Increased epochs
            'optimizer': 'Adam',
            'weight_decay': 1e-5,
            'momentum': 0.9,  # Not used with Adam, but kept for potential SGD use
            'scheduler_step': 5,
            'scheduler_gamma': 0.5
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_interval = 50  # Log every 50 batches

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=hparams_dict['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams_dict['batch_size'], drop_last=True)

    # Model, optimizer, scheduler, and loss
    model = ConvNet(dropout_rate=hparams_dict['dropout_rate']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams_dict['learning_rate'], weight_decay=hparams_dict['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams_dict['scheduler_step'], gamma=hparams_dict['scheduler_gamma'])
    criterion = nn.CrossEntropyLoss()

    # TensorBoard setup using the custom logger
    logger = TensorboardLogger(log_dir=log_dir, comment='MNIST', hparams_dict=hparams_dict)

    # Model graph
    images, _ = next(iter(train_loader))
    logger.log_graph(model, images.to(device))


    # Training loop
    global_step = 0
    viz_utils = VisualizationUtils()  # Instantiate VisualizationUtils

    for epoch in range(hparams_dict['epochs']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


            if batch_idx % log_interval == 0:
                # 1. Log basic metrics
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = correct / len(target)

                logger.log_scalar('Metrics/train_loss', loss.item(), global_step)
                logger.log_scalar('Metrics/train_accuracy', accuracy, global_step)
                logger.log_scalar('Learning/learning_rate', optimizer.param_groups[0]['lr'], global_step)

                # 2. Log parameter and gradient distributions
                for name, param in model.named_parameters():
                    logger.log_histogram(f'Parameters/{name}', param.data, global_step)
                    if param.grad is not None:
                        logger.log_histogram(f'Gradients/{name}', param.grad, global_step)
                        logger.log_scalar(f'Gradients/norm_{name}', param.grad.norm().item(), global_step)

                # 3. Log feature maps
                with torch.no_grad():  # No need to track gradients for visualization
                  feat1, feat2, _ = model.get_features()
                  logger.log_image('Features/conv1', viz_utils.visualize_feature_maps(feat1[0]), global_step)
                  logger.log_image('Features/conv2', viz_utils.visualize_feature_maps(feat2[0]), global_step)

                # 4. Log input images and filters (first batch of each epoch)
                if batch_idx == 0:
                    img_grid = utils.make_grid(data[:8])
                    logger.log_image('Images/input', img_grid, global_step)

                    filters = model.conv1.weight.data.cpu()
                    logger.log_image('Filters/conv1', utils.make_grid(filters, normalize=True, scale_each=True), global_step)

                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAccuracy: {accuracy:.4f}')
            global_step += 1



        # Validation at the end of each epoch
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = correct / len(test_loader.dataset)
        logger.log_scalar('Metrics/test_loss', test_loss, epoch)
        logger.log_scalar('Metrics/test_accuracy', accuracy, epoch)

        print(f'Epoch {epoch}: Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}')


        # Embedding visualization
        embeddings, labels = get_embeddings_and_labels(model, test_loader, device, num_samples=1000)

        # t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
        embeddings_2d = tsne.fit_transform(embeddings.numpy())

        # Create label images
        label_img = torch.zeros(labels.shape[0], 3, 28, 28)
        for i in range(labels.shape[0]):
            label_img[i] = test_dataset[i][0].repeat(3, 1, 1)  # Repeat for RGB channels

        logger.log_embedding(embeddings_2d, labels.tolist(), label_img, epoch, tag='embeddings')

        scheduler.step()

    # Log final hyperparameters and metrics
    logger.log_hparams(
        hparams_dict,
        {
            'hparam/accuracy': accuracy,
            'hparam/loss': test_loss
        }
    )

    logger.close()
    return accuracy

if __name__ == '__main__':
    # Single run with default hyperparameters
    train()

    # Example hyperparameter search (you could expand this)
    hparams_list = [
      {
          'batch_size': 32,
          'learning_rate': 0.001,
          'dropout_rate': 0.3,
          'epochs': 5,
          'optimizer': 'Adam',
          'weight_decay': 1e-5,
          'momentum': 0.9,
          'scheduler_step': 2,
          'scheduler_gamma': 0.5
        },
        {
            'batch_size': 64,
            'learning_rate': 0.0005,
            'dropout_rate': 0.5,
            'epochs': 5,
            'optimizer': 'Adam',
            'weight_decay': 1e-4,
            'momentum': 0.9,
            'scheduler_step': 3,
            'scheduler_gamma': 0.2
        },
    ]

    for hparams in hparams_list:
      print(f"\nTraining with hyperparameters: {hparams}")
      train(hparams_dict=hparams)