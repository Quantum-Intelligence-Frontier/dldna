import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd  # pandas import added

def visualize_layer_normalization(batch_size=2, seq_length=5, hidden_size=6, figure_width_percentage=0.7):
    """
    Visually rich and intuitive explanation of the Layer Normalization process.
    Uses various visualization elements to clearly show each step.

    Args:
        batch_size: Batch size.
        seq_length: Sequence length.
        hidden_size: Hidden layer size.
        figure_width_percentage: The ratio of the graph width to the total figure width (between 0 and 1).
    """

    # --- Setup ---
    np.random.seed(42)
    sns.set_style("whitegrid")
    # Set overall figure size
    plt.rcParams['figure.figsize'] = [18, 10]  # Default size
    plt.rcParams['font.size'] = 12

    # --- 1. Create Input Data ---
    x = np.random.randn(batch_size, seq_length, hidden_size) * 5

    # --- 2. Calculate Mean and Standard Deviation ---
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)

    # --- 3. Perform Normalization ---
    normalized = (x - mean) / (std + 1e-12)

    # --- 4. Apply Learnable Parameters ---
    gamma = np.ones(hidden_size) + np.random.randn(hidden_size) * 0.1
    beta = np.zeros(hidden_size) + np.random.randn(hidden_size) * 0.1
    scaled = gamma * normalized + beta

    # --- Visualization ---
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)

    # --- 4.1 Original Data Distribution (Histogram + KDE) ---
    ax1 = fig.add_subplot(gs[0, :2])
    sns.histplot(data=x.flatten(), bins=30, ax=ax1, color='skyblue', stat='density', kde=True,
                 line_kws={'linewidth': 3})
    ax1.set_title('1. Original Data Distribution', fontsize=16)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    # Width adjustment
    ax1.set_position([ax1.get_position().x0, ax1.get_position().y0,
                      ax1.get_position().width * figure_width_percentage, ax1.get_position().height])


    # --- 4.2 Normalized Data Distribution (Histogram + KDE) ---
    ax2 = fig.add_subplot(gs[0, 2:])
    sns.histplot(data=normalized.flatten(), bins=30, ax=ax2, color='lightgreen', stat='density', kde=True,
                 line_kws={'linewidth': 3})
    ax2.set_title('2. Normalized Data Distribution', fontsize=16)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    # Width adjustment
    ax2.set_position([ax2.get_position().x0, ax2.get_position().y0,
                      ax2.get_position().width * figure_width_percentage, ax2.get_position().height])


    # --- 4.3 Distribution after Scale and Shift
    ax_scaled = fig.add_subplot(gs[1, 1:3]) # Centered
    sns.histplot(data=scaled.flatten(), bins=30, ax=ax_scaled, color='coral', stat='density', kde=True,
                line_kws={'linewidth':3})
    ax_scaled.set_title('3. Scaled & Shifted Data Distribution', fontsize=16)
    ax_scaled.set_xlabel("Value")
    ax_scaled.set_ylabel("Density")
    # Width adjustment
    ax_scaled.set_position([ax_scaled.get_position().x0, ax_scaled.get_position().y0,
                            ax_scaled.get_position().width * figure_width_percentage, ax_scaled.get_position().height])



    # --- 4.4 Original Data Heatmap (First Batch) ---
    ax3 = fig.add_subplot(gs[2, 0])
    sns.heatmap(x[0], ax=ax3, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Value'}, linewidths=.5)
    ax3.set_title('Original Data (First Batch)', fontsize=14)
    ax3.set_xlabel('Hidden Dimension')
    ax3.set_ylabel('Sequence Position')

    # --- 4.5 Normalized Data Heatmap (First Batch) ---
    ax4 = fig.add_subplot(gs[2, 1])
    sns.heatmap(normalized[0], ax=ax4, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Value'}, linewidths=.5)
    ax4.set_title('Normalized Data (First Batch)', fontsize=14)
    ax4.set_xlabel('Hidden Dimension')
    ax4.set_ylabel('Sequence Position')

    # --- 4.6  Scaled/Shifted Data Heatmap (First Batch)---
    ax5 = fig.add_subplot(gs[2, 2])
    sns.heatmap(scaled[0], ax=ax5, cmap='viridis', annot=True, fmt='.2f', cbar_kws={'label': 'Value'}, linewidths=.5)
    ax5.set_title('Scaled & Shifted Data (First Batch)', fontsize=14)
    ax5.set_xlabel('Hidden Dimension')
    ax5.set_ylabel('Sequence Position')

    # --- 4.7 Gamma/Beta Value Visualization (Bar Graph) ---
    ax6 = fig.add_subplot(gs[2, 3])
    gamma_beta_df = pd.DataFrame({'Gamma': gamma, 'Beta': beta})
    gamma_beta_df.plot(kind='bar', ax=ax6, color=['skyblue', 'lightgreen'], rot=0)
    ax6.set_title('Gamma and Beta Values', fontsize=14)
    ax6.set_xlabel('Hidden Dimension')
    ax6.set_ylabel('Value')
    ax6.legend(loc="upper right")

    # --- Overall Title ---
    fig.suptitle('Layer Normalization Visualization', fontsize=20, fontweight='bold')

    plt.show()

    # --- Output Results (Console) ---
    print("=" * 40)
    print("Input Data Shape:", x.shape)
    print("Mean Shape:", mean.shape)
    print("Standard Deviation Shape:", std.shape)
    print("Normalized Data Shape:", normalized.shape)
    print("Gamma (Scale) Values:\n", gamma)
    print("Beta (Shift) Values:\n", beta)
    print("Scaled & Shifted Data Shape:", scaled.shape)
    print("=" * 40)

