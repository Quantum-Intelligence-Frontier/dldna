import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

class ConvVisualizer:
    """Convolution operation visualization class"""

    def __init__(self):
        """Initialize matrix and visualization settings"""

        # Input matrix (6x6) - made it a bit more interesting
        self.in_mat = np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30, 31],
            [32, 33, 34, 35, 36, 37]
        ])

        # Sobel filter (for edge detection)
        self.kern = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ])

        # Output matrix initialization
        self.out_vals = np.zeros((self.in_mat.shape[0]-2,
                                  self.in_mat.shape[1]-2))
        self.out_colors = np.zeros((self.in_mat.shape[0]-2,
                                    self.in_mat.shape[1]-2, 3))

        # --- Setup the plot ---
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            1, 3, figsize=(18, 6),
            gridspec_kw={'width_ratios': [6, 3, 4]}
        )
        plt.close() # Prevents initial display

        # --- Styling ---
        self.cmap = "RdYlGn"  # Red-Yellow-Green colormap

    def _add_labels(self, ax, mat, highlight=None):
        """Display matrix values, with optional highlighting and dynamic colors."""
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "black"  # Default text color
                fontsize = 14

                if highlight and (i, j) in highlight:
                    color = "white"  # Highlighted text color
                    fontsize = 16

                # Use formatted string for consistent display
                ax.text(j, i, f"{val:.0f}", ha='center', va='center',
                        fontsize=fontsize, fontweight='bold', color=color)


    def _animate(self, frame):
        """Perform animation for each frame."""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Calculate current position
        i = frame // (self.in_mat.shape[1] - 2)
        j = frame % (self.in_mat.shape[1] - 2)

        # --- Input Matrix Visualization ---
        self.ax1.imshow(self.in_mat, cmap=self.cmap, vmin=1, vmax=37)  # Use the colormap
        self.ax1.set_title('Input Matrix',  fontsize=18, fontweight='bold', pad=15)
        highlight_cells = [(row, col) for row in range(i, i+3) for col in range(j, j+3)]
        self._add_labels(self.ax1, self.in_mat, highlight=highlight_cells)

        # Highlight the current 3x3 region.
        rect = plt.Rectangle((j - 0.5, i - 0.5), 3, 3, linewidth=4, edgecolor='#d62728', facecolor='none') # Darker Red
        self.ax1.add_patch(rect)


        # --- Kernel Visualization ---
        self.ax2.imshow(self.kern, cmap=self.cmap, vmin=-2, vmax=2)  # Fixed color range and colormap
        self.ax2.set_title('Kernel',  fontsize=18, fontweight='bold', pad=15)
        self._add_labels(self.ax2, self.kern)

        # Kernel grid (cleaner way)
        self.ax2.set_xticks(np.arange(-.5, 3, 1))
        self.ax2.set_yticks(np.arange(-.5, 3, 1))
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax2.tick_params(length=0)  # Hide ticks
        for spine in self.ax2.spines.values():  #Optional, for a box around
             spine.set_visible(True)
             spine.set_linewidth(2)
             spine.set_edgecolor('black')

        # --- Output Matrix Visualization ---

        if i < self.in_mat.shape[0]-2 and j < self.in_mat.shape[1]-2:
            curr_reg = self.in_mat[i:i+3, j:j+3]
            result = np.sum(curr_reg * self.kern)
            self.out_vals[i, j] = result

            # Normalize and get color from colormap
            norm_res = (result + 8) / 16
            norm_res = np.clip(norm_res, 0, 1)  # Ensure it's within 0-1
            self.out_colors[i, j] = plt.get_cmap(self.cmap)(norm_res)[:3]

        self.ax3.imshow(self.out_colors,  vmin=-8, vmax=8) # Consistent color range
        self.ax3.set_title('Output Matrix', fontsize=18, fontweight='bold', pad=15)
        self._add_labels(self.ax3, self.out_vals)


        # --- Grids and Ticks (Cleaned Up) ---
        for ax in [self.ax1, self.ax3]:
            ax.set_xticks(np.arange(-.5, ax.get_images()[0].get_array().shape[1], 1))
            ax.set_yticks(np.arange(-.5, ax.get_images()[0].get_array().shape[0], 1))
            ax.set_xticklabels([])  # Remove tick labels
            ax.set_yticklabels([])  # Remove tick labels
            ax.tick_params(length=0) # Hide ticks
            ax.grid(color='lightgray', linestyle='-', linewidth=1)  # Lighter grid

        # Add a title to the animation
        self.fig.suptitle("2D Convolution with Sobel Filter", fontsize=20, fontweight='bold', y=0.98)


    def create_anim(self):
        """Create convolution animation"""
        anim = FuncAnimation(
            self.fig,
            self._animate,
            frames=(self.in_mat.shape[0]-2) * (self.in_mat.shape[1]-2),
            interval=500,  # milliseconds
            repeat=True
        )

        return HTML(anim.to_jshtml())

def create_conv_animation():
    """Convenience function to create and return animation"""
    viz = ConvVisualizer()
    return viz.create_anim()