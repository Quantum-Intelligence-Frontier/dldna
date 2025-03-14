import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def plot_results(results: Dict[str, Any]) -> None:
    """Visualizes the training results (accuracy)."""
    train_acc = results['train_scores']
    test_acc = results['test_scores']
    epochs = range(len(train_acc))

    plt.figure(figsize=(5, 3))
    plt.title('Accuracy', fontsize=10)
    plt.xlabel('Epochs', fontsize=10)

    sns.lineplot(x=epochs, y=train_acc, color="C0", label='Train')
    sns.lineplot(x=epochs, y=test_acc, color="C1", label='Test')

    plt.legend()
    plt.show()

def create_results_table(results_dict: Dict[str, Dict[str, float]]) -> None:
    """Prints the results as a Markdown table."""
    print("Model | Accuracy (%) | Final Loss | Time (s)")
    print("-- | -- | -- | --")
    for model_name, metrics in results_dict.items():
        print(f"{model_name} | {metrics['accuracy']:.1f} | "
              f"{metrics['loss']:.2f} | {metrics['time']:.1f}")