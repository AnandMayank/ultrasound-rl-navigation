# Segmentation Examples

This folder contains example results from the ResNet U-Net segmentation model trained on abdominal ultrasound images.

## File Naming Convention

- `original_X.png` - Original ultrasound images
- `mask_X.png` - Generated binary segmentation masks
- `overlay_X.png` - Overlay visualization (original + mask)

Where X is the example number (0-9) or specific cropped image identifiers (e.g., cropped_11, cropped_25, cropped_33, cropped_15).

### Enhanced Examples

The folder includes additional high-quality segmentation examples:
- `*_cropped_11.*` - Clear abdominal region with well-defined boundaries
- `*_cropped_25.*` - Complex anatomical structure segmentation
- `*_cropped_33.*` - High-contrast segmentation example
- `*_cropped_15.*` - Test set example showing generalization

## Model Performance

The ResNet U-Net model demonstrates:

### Strengths
- Accurate boundary detection of abdominal regions
- Robust performance across different image qualities
- Effective handling of ultrasound artifacts and noise
- Consistent segmentation of anatomical structures

### Key Features
- **Architecture**: ResNet18 backbone with U-Net decoder
- **Loss Function**: Combined Binary Cross-Entropy and Dice Loss
- **Input Size**: 256x256 grayscale images
- **Output**: Binary segmentation masks

### Training Details
- Trained on abdominal ultrasound dataset
- Data augmentation for improved generalization
- Early stopping based on validation loss
- Best model selected based on validation performance

## Usage

These segmentation results serve as input for the navigation system, where the centers of the segmented regions become targets for the RL agent to navigate to.

The segmentation quality directly impacts navigation performance, making accurate segmentation crucial for the overall system success.
