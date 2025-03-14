# chapter03.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def simulate_dice_roll():
    # Dice roll simulation
    rolls = np.random.randint(1, 7, 1000)
    unique, counts = np.unique(rolls, return_counts=True)

    plt.bar(unique, counts/1000)
    plt.xlabel('Dice Value')
    plt.ylabel('Probability')
    plt.title('Dice Roll Probability Distribution')
    plt.show()

def plot_normal_distribution():
    x = np.linspace(-3, 3, 100)
    y = norm.pdf(x, 0, 1)

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Standard Normal Distribution')
    plt.show()

def calculate_dice_expected_value():
    outcomes = np.arange(1, 7)
    probabilities = np.ones(6) / 6  # Each outcome has probability 1/6

    expected_value = np.sum(outcomes * probabilities)
    print(f"Expected value of dice roll: {expected_value}")
