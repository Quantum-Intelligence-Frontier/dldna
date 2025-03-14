# information_theory.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def calculate_entropy():
    # Calculate entropy for different probability distributions
    p1 = np.array([0.5, 0.5])  # Fair coin
    p2 = np.array([0.9, 0.1])  # Biased coin
    p3 = np.array([0.25, 0.25, 0.25, 0.25])  # Fair die

    H1 = entropy(p1)
    H2 = entropy(p2)
    H3 = entropy(p3)

    print(f"Entropy of fair coin: {H1:.2f}")
    print(f"Entropy of biased coin: {H2:.2f}")
    print(f"Entropy of fair die: {H3:.2f}")

    # Plot entropy for binary distribution
    p = np.linspace(0.01, 0.99, 100)
    H = -p * np.log2(p) - (1-p) * np.log2(1-p)

    plt.plot(p, H)
    plt.xlabel('Probability of heads')
    plt.ylabel('Entropy (bits)')
    plt.title('Entropy of a Binary Distribution')
    plt.show()

def mutual_information_example():
    # Create joint probability distribution
    joint_prob = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    # Calculate marginal probabilities
    p_x = np.sum(joint_prob, axis=1)
    p_y = np.sum(joint_prob, axis=0)
    
    # Calculate mutual information
    mi = np.sum(joint_prob * np.log2(joint_prob / (p_x[:, np.newaxis] * p_y)))
    
    print(f"Mutual Information: {mi:.4f}")

    # Visualize joint distribution
    plt.imshow(joint_prob, cmap='viridis')
    plt.colorbar(label='Joint Probability')
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.title('Joint Probability Distribution')
    plt.show()

def kl_divergence_example():
    # Define two probability distributions
    p = np.array([0.3, 0.7])
    q = np.array([0.5, 0.5])

    # Calculate KL divergence
    kl_pq = entropy(p, q)
    kl_qp = entropy(q, p)

    print(f"KL(P||Q): {kl_pq:.4f}")
    print(f"KL(Q||P): {kl_qp:.4f}")

    # Visualize distributions
    x = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, p, width, label='P')
    ax.bar(x + width/2, q, width, label='Q')

    ax.set_ylabel('Probability')
    ax.set_title('Probability Distributions P and Q')
    ax.set_xticks(x)
    ax.set_xticklabels(['Event 1', 'Event 2'])
    ax.legend()

    plt.show()
