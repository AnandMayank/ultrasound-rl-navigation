import torch
import numpy as np
import os
import glob
import json
from PIL import Image

from core.segmentation_model import ResNetUNet
from core.navigation_agent import NavigationAgent
from core.navigation_environment import NavigationEnvironment
from core.utils import find_center, visualize_segmentation, visualize_navigation
from train_segmentation import train_segmentation_model
from train_navigation import train_navigation_agent, evaluate_agent


def segment_images(model_path, image_dir, save_dir):
    """Segment images and find centers"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNetUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    os.makedirs(save_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    centers_dict = {}
    
    for i, image_path in enumerate(image_files):
        image = Image.open(image_path).convert('L')
        image_resized = image.resize((256, 256))
        
        image_tensor = torch.FloatTensor(np.array(image_resized) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mask_tensor = model(image_tensor)
        
        mask = mask_tensor.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        center = find_center(mask)
        
        filename = os.path.basename(image_path)
        centers_dict[filename] = center
        
        if i < 10:
            visualize_segmentation(
                np.array(image_resized),
                mask,
                center,
                os.path.join(save_dir, f"segmentation_{i}.png")
            )
    
    with open(os.path.join(save_dir, "centers.json"), "w") as f:
        json.dump(centers_dict, f, indent=4)
    
    return centers_dict


def run_complete_pipeline():
    """Run the complete pipeline"""
    print("Starting Ultrasound Image Segmentation and RL Navigation Pipeline")
    
    # Configuration
    config = {
        'data_dir': './abdominal_US',
        'image_dir': './Abdomen_simulation',
        'results_dir': './pipeline_results',
        'segmentation_epochs': 20,
        'navigation_episodes': 500
    }
    
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Step 1: Train segmentation model
    print("\n1. Training segmentation model...")
    seg_save_dir = os.path.join(config['results_dir'], 'segmentation')
    
    if os.path.exists(os.path.join(seg_save_dir, 'best_model.pth')):
        print("Segmentation model already exists, skipping training...")
        model_path = os.path.join(seg_save_dir, 'best_model.pth')
    else:
        model, train_losses, val_losses = train_segmentation_model(
            data_dir=config['data_dir'],
            save_dir=seg_save_dir,
            num_epochs=config['segmentation_epochs']
        )
        model_path = os.path.join(seg_save_dir, 'best_model.pth')
    
    # Step 2: Segment images and find centers
    print("\n2. Segmenting images and finding centers...")
    centers_dict = segment_images(
        model_path=model_path,
        image_dir=config['image_dir'],
        save_dir=os.path.join(config['results_dir'], 'segmentation_results')
    )
    
    # Step 3: Train navigation agent
    print("\n3. Training navigation agent...")
    nav_save_dir = os.path.join(config['results_dir'], 'navigation')
    
    image_files = glob.glob(os.path.join(config['image_dir'], "*.png"))
    
    if len(image_files) == 0:
        print("No images found for navigation training!")
        return
    
    np.random.shuffle(image_files)
    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]
    
    agent, metrics = train_navigation_agent(
        image_files=train_files,
        centers_dict=centers_dict,
        save_dir=nav_save_dir,
        num_episodes=config['navigation_episodes']
    )
    
    # Step 4: Evaluate the complete system
    print("\n4. Evaluating the complete system...")
    test_env = NavigationEnvironment(test_files, centers_dict)
    eval_metrics = evaluate_agent(
        env=test_env,
        agent=agent,
        num_episodes=20,
        save_dir=os.path.join(config['results_dir'], 'evaluation')
    )
    
    # Step 5: Generate summary report
    print("\n5. Generating summary report...")
    summary = {
        'config': config,
        'segmentation_model': model_path,
        'navigation_agent': os.path.join(nav_save_dir, 'agent_final.pt'),
        'evaluation_metrics': eval_metrics,
        'total_images': len(image_files),
        'training_images': len(train_files),
        'test_images': len(test_files)
    }
    
    with open(os.path.join(config['results_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nPipeline completed successfully!")
    print(f"Results saved to: {config['results_dir']}")
    print(f"Success rate: {eval_metrics['success_rate']:.2%}")
    print(f"Average final distance: {eval_metrics['avg_final_distance']:.2f}")


def demo_single_image(image_path, segmentation_model_path, navigation_model_path):
    """Demo on a single image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    seg_model = ResNetUNet()
    seg_model.load_state_dict(torch.load(segmentation_model_path, map_location=device))
    seg_model.to(device)
    seg_model.eval()
    
    nav_agent = NavigationAgent(num_actions=5, device=device)
    nav_agent.load(navigation_model_path)
    
    # Process image
    image = Image.open(image_path).convert('L')
    image_resized = image.resize((256, 256))
    
    # Segment
    image_tensor = torch.FloatTensor(np.array(image_resized) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_tensor = seg_model(image_tensor)
    
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    center = find_center(mask)
    
    # Navigate
    centers_dict = {os.path.basename(image_path): center}
    env = NavigationEnvironment([image_path], centers_dict)
    
    state = env.reset()
    positions = [env.current_position]
    
    for step in range(50):
        action = nav_agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        positions.append(env.current_position)
        state = next_state
        
        if done:
            break
    
    # Visualize results
    visualize_segmentation(np.array(image_resized), mask, center, "demo_segmentation.png")
    visualize_navigation(np.array(image), env.current_position, center, save_path="demo_navigation.png")
    
    print(f"Demo completed! Final distance to target: {info['distance']:.2f}")
    return info['distance']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ultrasound Image Segmentation and RL Navigation')
    parser.add_argument('--mode', choices=['pipeline', 'demo'], default='pipeline',
                       help='Run complete pipeline or single image demo')
    parser.add_argument('--image', type=str, help='Path to single image for demo mode')
    parser.add_argument('--seg_model', type=str, default='./results/trained_models/simple_resnet_unet_best.pth',
                       help='Path to segmentation model')
    parser.add_argument('--nav_model', type=str, default='./results/trained_models/agent_final.pt',
                       help='Path to navigation model')

    args = parser.parse_args()

    if args.mode == 'pipeline':
        run_complete_pipeline()
    elif args.mode == 'demo':
        if not args.image:
            print("Please provide --image path for demo mode")
        else:
            demo_single_image(args.image, args.seg_model, args.nav_model)
