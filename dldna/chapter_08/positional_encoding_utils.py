import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_sinusoidal_features():
    # Settings
    plt.figure(figsize=(0.6 * 15, 0.6 * 12))  # Reduce figure size to 80%
    sns.set_style("whitegrid")
    
    # 1. Different frequencies of sine functions
    positions = np.arange(100)
    plt.subplot(2, 2, 1)
    wavelengths = [10, 20, 40, 80]
    for wavelength in wavelengths:
        sin = np.sin(positions * (1/wavelength))
        sns.lineplot(x=positions, y=sin, label=f'wavelength={wavelength}')
    plt.title('1. Different Frequencies of Sine Functions', pad=20)
    plt.legend()
    
    # 2. Sine-cosine pair
    plt.subplot(2, 2, 2)
    wavelength = 20
    sin = np.sin(positions * (1/wavelength))
    cos = np.cos(positions * (1/wavelength))
    sns.lineplot(x=positions, y=sin, label='sine')
    sns.lineplot(x=positions, y=cos, label='cosine')
    plt.title(f'2. Sine-Cosine Pair (wavelength={wavelength})', pad=20)
    plt.legend()

    # 3. Position shifts in sinusoidal encoding
    plt.subplot(2, 2, 3)
    positions = np.arange(20)
    wavelength = 100  # Set to a longer wavelength
    distances = [5, 10, 15]

    # Sine function of the reference position
    pos1 = np.sin(positions * (1/wavelength))
    sns.lineplot(x=positions, y=pos1, label='position', color='blue')

    # Sine functions shifted by each distance
    colors = ['red', 'green', 'purple']
    for dist, color in zip(distances, colors):
        shifted_pos = np.sin((positions + dist) * (1/wavelength))
        sns.lineplot(x=positions, y=shifted_pos, 
                    linestyle='--', 
                    label=f'position+{dist}',
                    color=color)

    plt.title('3. Position Shifts in Sinusoidal Encoding', pad=20)
    plt.legend()
    plt.grid(True)  
        
    # 4. Actual positional encoding heatmap
    plt.subplot(2, 2, 4)
    seq_length = 20
    d_model = 32
    pe = positional_encoding(seq_length, d_model)
    
    sns.heatmap(pe, cmap='coolwarm', center=0, cbar_kws={'label': 'Encoding Value'})
    plt.title('4. Positional Encoding Matrix Heatmap', pad=20)
    plt.xlabel('Encoding Dimension')
    plt.ylabel('Sequence Position')
    
    plt.tight_layout()
    plt.show()

def positional_encoding(seq_length, d_model):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe



def show_positional_periods(d_model=512):
    # Dimension indices
    i_values = [0, d_model//4, d_model//2 - 1]  # First, middle, and last dimensions

    # Calculate periods
    periods = []
    for i in i_values:
        wavelength = np.power(10000, (2*i)/d_model)
        periods.append(wavelength)

    print("1. Periods of positional encoding:")
    print(f"First dimension (i=0): {periods[0]:.2f}")
    print(f"Middle dimension (i={d_model//4}): {periods[1]:.2f}")
    print(f"Last dimension (i={d_model//2 - 1}): {periods[2]:.2f}")

    print("\n2. Positional encoding formula values (10000^(2i/d_model)):")
    for i in i_values:
        wavelength = np.power(10000, (2*i)/d_model)
        print(f"i={i:3d}: {wavelength:.10f}")

    # Calculate div_term (used in the actual implementation)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    print("\n3. Actual div_term values (first/middle/last):")
    print(f"First (i=0): {div_term[0]:.10f}")
    print(f"Middle (i={d_model//4}): {div_term[d_model//4]:.10f}")
    print(f"Last (i={d_model//2 - 1}): {div_term[-1]:.10f}")


def show_positional_encoding_steps(seq_length=4, d_model=512):
    """Shows the step-by-step calculation of positional encoding."""

    print("Step-by-step calculation of positional encoding")
    print("=" * 50)

    # Step 1: Create position indices
    print("1. Create position indices")
    print("-" * 50)
    print("Source code:")
    print("position = np.arange(seq_length)[:, np.newaxis]")

    position = np.arange(seq_length)[:, np.newaxis]
    print("\nResult (first 5 positions):")
    print(position[:5])
    print("shape:", position.shape)
    print("\n")

    # Step 2: Calculate periods for each dimension
    print("2. Calculate periods for each dimension")
    print("-" * 50)
    print("Source code:")
    print("div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))")

    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    print("\nResult:")
    print("Periods for the first 5 dimensions:", div_term[:5])
    print("Periods for the last 5 dimensions:", div_term[-5:])
    print("shape:", div_term.shape)
    print("\n")

    # Step 3: Apply sine/cosine
    print("3. Apply sine/cosine")
    print("-" * 50)
    print("Source code:")
    print("pe = np.zeros((seq_length, d_model))")
    print("pe[:, 0::2] = np.sin(position * div_term)")
    print("pe[:, 1::2] = np.cos(position * div_term)")

    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    print("\nResult (first 3 positions, first 8 dimensions):")
    print(pe[:3, :8])
    print("shape:", pe.shape)


if __name__ == "__main__":
    visualize_sinusoidal_features()
    show_positional_periods()