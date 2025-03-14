import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation
import requests
from PIL import Image
from io import BytesIO

def load_image_from_url(url):
    """Downloads an image from a URL and converts it to grayscale."""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('L')
    return np.array(img)

def show_filter_effects(image_url):
    """Visualizes the effects of various filters."""
    # Load image
    image = load_image_from_url(image_url)
    
    # Resize image
    image = cv2.resize(image, (400, 400))
    
    # Define various filters
    filters = {
        'Gaussian Blur': cv2.getGaussianKernel(3, 1) @ cv2.getGaussianKernel(3, 1).T,
        'Sharpen': np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]),
        'Edge Detection': np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]),
        'Emboss': np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ]),
        'Sobel X': np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Display original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Display the result of applying each filter
    row = 0
    col = 1
    for name, kernel in filters.items():
        filtered = cv2.filter2D(image, -1, kernel)
        axes[row, col].imshow(filtered, cmap='gray')
        axes[row, col].set_title(name)
        axes[row, col].axis('off')
        
        col += 1
        if col > 2:
            col = 0
            row += 1
    
    plt.tight_layout()
    plt.show()

def create_convolution_animation(image_url):
    """Visualizes the convolution operation process with an animation."""
    # Load and resize image
    image = load_image_from_url(image_url)
    image = cv2.resize(image, (8, 8))  # Resize to a small size for animation
    
    # Example kernel (Gaussian blur kernel)
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.0
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.close()  # Close the figure to prevent it from displaying before animation

    def update(frame):
        ax.clear()
        i, j = frame // image.shape[1], frame % image.shape[1]
        
        # Highlight the current convolution position
        highlighted = np.zeros_like(image)
        if i < image.shape[0] - 2 and j < image.shape[1] - 2:
            highlighted[i:i+3, j:j+3] = 1
        
        # Display the image and highlighted area
        ax.imshow(image, cmap='gray')
        ax.imshow(highlighted, cmap='Reds', alpha=0.3)  # Use Reds colormap for highlighting
        ax.set_title(f'Convolution Step: {frame+1}')
        
    anim = FuncAnimation(fig, update, 
                        frames=image.shape[0] * image.shape[1],
                        interval=500, repeat=False)
    return anim

def visualize_convolution_step(input_matrix, kernel):
    """Visualizes a single step of the convolution operation."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display input matrix
    ax1.imshow(input_matrix, cmap='gray')
    ax1.set_title('Input Matrix')
    
    # Display kernel
    ax2.imshow(kernel, cmap='gray')
    ax2.set_title('Kernel')
    
    # Display convolution result
    result = cv2.filter2D(input_matrix.astype(np.float32), -1, kernel)
    ax3.imshow(result, cmap='gray')
    ax3.set_title('Convolution Result')
    
    plt.tight_layout()
    plt.show()
    
    return result

def apply_multiple_filters(image, kernels):
    """Applies multiple filters sequentially."""
    results = []
    for name, kernel in kernels.items():
        filtered = cv2.filter2D(image, -1, kernel)
        results.append((name, filtered))
    return results


# Default image URL for testing
DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg"

if __name__ == "__main__":
    # Test basic filter effects
    show_filter_effects(DEFAULT_IMAGE_URL)
    
    # Test convolution animation
    anim = create_convolution_animation(DEFAULT_IMAGE_URL)
    plt.show()