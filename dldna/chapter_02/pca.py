import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_pca():

    # Generate complex 2D data
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    x = t * np.cos(t)
    y = t * np.sin(t) + 0.5 * np.random.randn(500)
    X = np.column_stack((x, y))

    # Perform PCA
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)

    # Visualize the results
    plt.figure(figsize=(12, 5))

    # Original data
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7, c=t, cmap='viridis')
    plt.title("Original 2D Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label='t')

    # PCA-transformed data
    plt.subplot(122)
    plt.scatter(X_pca, np.zeros_like(X_pca), alpha=0.7, c=t, cmap='viridis')
    plt.title("PCA-transformed 1D Data")
    plt.xlabel("First Principal Component")
    plt.yticks([])
    plt.colorbar(label='t')

    plt.tight_layout()
    plt.show()

    print(f"Explained variance ratio: {pca.explained_variance_ratio_[0]:.4f}")