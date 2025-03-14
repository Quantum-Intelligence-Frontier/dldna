import os
import datetime
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class VisualizationUtils:
    """
    TensorBoard visualization utilities.
    This class provides static methods for generating visualizations
    suitable for TensorBoard, such as image grids, feature map visualizations,
    and confusion matrices.  It aims to keep visualization logic separate
    from model training logic.
    """

    @staticmethod
    def make_grid_with_labels(images, labels, num_images=8):
        """Creates a grid of images with corresponding labels."""
        # Ensure we don't try to display more images than available
        num_images = min(num_images, len(images))

        fig, axes = plt.subplots(nrows=int(np.sqrt(num_images)), ncols=int(np.sqrt(num_images)), figsize=(8, 8))
        for idx, ax in enumerate(axes.flat):
            if idx < num_images:
                img = images[idx].squeeze()  # Remove channel dimension if it's 1
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Label: {labels[idx]}')
                ax.axis('off')
            else:
                ax.axis('off')  # Hide extra subplots if num_images is not a perfect square

        return fig


    @staticmethod
    def visualize_feature_maps(feature_maps, num_features=8):
        """Visualizes feature maps as a grid."""
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = feature_maps.detach().cpu()

        # Ensure we don't try to display more feature maps than we have.
        num_features = min(num_features, feature_maps.size(0))

        # make_grid expects (C, H, W), and we add a channel dimension.
        grid = torchvision.utils.make_grid(
            feature_maps[:num_features].unsqueeze(1),
            normalize=True,
            nrow=int(np.sqrt(num_features))  # Control the number of rows/columns
        )
        return grid

    @staticmethod
    def plot_confusion_matrix(cm, class_names):
        """Plots a confusion matrix."""
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig


def get_embeddings_and_labels(model, loader, device, num_samples=1000):
    """Extracts embeddings and labels from a data loader."""
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            # Stop when we reach the desired number of samples
            if len(embeddings) * data.size(0) >= num_samples:
                break
            data = data.to(device)
            embedding = model.get_embeddings(data)
            embeddings.append(embedding.cpu())
            labels.append(target)
    return torch.cat(embeddings), torch.cat(labels)




class TensorboardLogger:
    """
    Handles TensorBoard logging.  This class encapsulates the SummaryWriter
    and provides convenient methods for logging various data types during
    model training.  It simplifies the process of logging scalars, histograms,
    images, embeddings, and hyperparameters.
    """

    def __init__(self, log_dir=None, comment='', hparams_dict=None):
        """
        Initializes the TensorboardLogger.

        Args:
            log_dir (str, optional): Directory to save logs.  If None, a
                directory is created based on the current time.
            comment (str, optional):  A comment to append to the log directory name.
            hparams_dict (dict, optional): Initial hyperparameters to log.
        """
        if log_dir is None:
            current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_dir = os.path.join('runs', f'experiment_{current_time}_{comment}')
        self.writer = SummaryWriter(log_dir=log_dir)
        if hparams_dict:
            self.writer.add_hparams(hparams_dict, {'hparam/dummy': 0}) # Add a dummy metric

    def log_scalar(self, tag, value, step):
        """Logs a scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        """Logs a histogram of values."""
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag, img_tensor, step):
        """Logs an image."""
        self.writer.add_image(tag, img_tensor, step)

    def log_figure(self, tag, figure, step):
        """Logs a matplotlib figure."""
        self.writer.add_figure(tag, figure, step)

    def log_embedding(self, embeddings, labels, label_images, step, tag='default'):
        """Logs embeddings for visualization with the Embedding Projector."""
        self.writer.add_embedding(embeddings, metadata=labels, label_img=label_images, global_step=step, tag=tag)

    def log_hparams(self, hparams_dict, metrics_dict):
         """Logs hyperparameters and corresponding metrics."""
         self.writer.add_hparams(hparam_dict, metrics_dict)

    def log_graph(self, model, input_tensor):
        """Logs the model graph."""
        self.writer.add_graph(model, input_tensor)

    def close(self):
        """Closes the SummaryWriter."""
        self.writer.close()