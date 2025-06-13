import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch


def find_center(mask):
    """Find the center of the segmented region"""
    if np.sum(mask) > 0:
        y_indices, x_indices = np.where(mask)
        center_x = int(np.mean(x_indices))
        center_y = int(np.mean(y_indices))
        return (center_x, center_y)
    else:
        return (mask.shape[1] // 2, mask.shape[0] // 2)


def visualize_segmentation(image, mask, center=None, save_path=None):
    """Visualize segmentation results with overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(mask, alpha=0.5, cmap='Reds')
    if center:
        axes[2].plot(center[0], center[1], 'b*', markersize=15, label='Center')
        axes[2].legend()
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_navigation(image, position, center, view_size=(64, 64), save_path=None):
    """Visualize navigation state"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.imshow(image, cmap='gray')
    
    x, y = position
    view_w, view_h = view_size
    
    rect = plt.Rectangle((x - view_w//2, y - view_h//2), view_w, view_h, 
                        linewidth=2, edgecolor='blue', facecolor='none', label='Agent View')
    ax.add_patch(rect)
    
    ax.plot(center[0], center[1], 'r*', markersize=15, label='Target Center')
    ax.plot(x, y, 'bo', markersize=8, label='Agent Position')
    
    ax.set_title('Navigation State')
    ax.legend()
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_navigation_gif(frames, save_path, duration=200):
    """Create GIF from navigation frames"""
    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )


def plot_training_metrics(metrics, save_path=None):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    episodes = range(len(metrics['episode_rewards']))
    
    axes[0, 0].plot(episodes, metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    axes[0, 1].plot(episodes, metrics['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    
    axes[0, 2].plot(episodes, metrics['final_distances'])
    axes[0, 2].set_title('Final Distance to Target')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Distance')
    
    axes[1, 0].plot(episodes, metrics['oscillation_counts'])
    axes[1, 0].set_title('Oscillation Counts')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Count')
    
    if 'training_losses' in metrics:
        axes[1, 1].plot(metrics['training_losses'])
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
    
    window_size = min(50, len(metrics['episode_rewards']))
    if window_size > 1:
        moving_avg = np.convolve(metrics['episode_rewards'], 
                               np.ones(window_size)/window_size, mode='valid')
        axes[1, 2].plot(range(window_size-1, len(episodes)), moving_avg)
        axes[1, 2].set_title(f'Reward Moving Average (window={window_size})')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Average Reward')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image for model input"""
    image = Image.open(image_path).convert('L')
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return torch.FloatTensor(image_array).unsqueeze(0).unsqueeze(0)


def postprocess_mask(mask_tensor, threshold=0.5):
    """Postprocess segmentation mask"""
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    return mask
