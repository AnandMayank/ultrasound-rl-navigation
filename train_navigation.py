import torch
import numpy as np
import random
import json
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.navigation_agent import NavigationAgent
from core.navigation_environment import NavigationEnvironment
from core.utils import plot_training_metrics, visualize_navigation


def train_navigation_agent(image_files, centers_dict, save_dir, num_episodes=500, 
                          save_freq=50, render_freq=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "renders"), exist_ok=True)
    
    env = NavigationEnvironment(image_files, centers_dict, view_size=(64, 64), max_steps=100)
    agent = NavigationAgent(
        num_actions=env.num_actions,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=10,
        device=device
    )
    
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'final_distances': [],
        'oscillation_counts': [],
        'training_losses': []
    }
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        oscillation_count = 0
        
        for step in range(env.max_steps):
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                metrics['training_losses'].append(loss)
            
            state = next_state
            total_reward += reward
            
            if env._is_oscillating():
                oscillation_count += 1
            
            if done:
                break
        
        metrics['episode_rewards'].append(total_reward)
        metrics['episode_lengths'].append(step + 1)
        metrics['final_distances'].append(info['distance'])
        metrics['oscillation_counts'].append(oscillation_count)
        
        if (episode + 1) % save_freq == 0:
            agent.save(os.path.join(save_dir, f"agent_episode_{episode+1}.pt"))
        
        if (episode + 1) % render_freq == 0:
            render_episode(env, agent, os.path.join(save_dir, "renders", f"episode_{episode+1}.png"))
        
        if episode % 100 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-100:])
            avg_distance = np.mean(metrics['final_distances'][-100:])
            print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, Avg Final Distance: {avg_distance:.2f}")
    
    agent.save(os.path.join(save_dir, "agent_final.pt"))
    
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    plot_training_metrics(metrics, os.path.join(save_dir, "metrics.png"))
    
    return agent, metrics


def render_episode(env, agent, save_path):
    """Render a single episode for visualization"""
    state = env.reset()
    
    for step in range(env.max_steps):
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        if step == 0 or done or step == env.max_steps - 1:
            fig = visualize_navigation(
                np.array(env.current_image),
                env.current_position,
                env.current_center,
                env.view_size
            )
            
            step_save_path = save_path.replace('.png', f'_step_{step}.png')
            plt.savefig(step_save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        state = next_state
        
        if done:
            break


def evaluate_agent(env, agent, num_episodes=20, save_dir=None):
    """Evaluate trained agent"""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    total_rewards = []
    final_distances = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(env.max_steps):
            action = agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        total_rewards.append(total_reward)
        final_distances.append(info['distance'])
        
        if info['distance'] < 20:
            success_count += 1
        
        if save_dir and episode < 5:
            render_episode(env, agent, os.path.join(save_dir, f"eval_episode_{episode}.png"))
    
    metrics = {
        'avg_reward': np.mean(total_rewards),
        'avg_final_distance': np.mean(final_distances),
        'success_rate': success_count / num_episodes,
        'total_episodes': num_episodes
    }
    
    print(f"Evaluation Results:")
    print(f"Average Reward: {metrics['avg_reward']:.2f}")
    print(f"Average Final Distance: {metrics['avg_final_distance']:.2f}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    
    if save_dir:
        with open(os.path.join(save_dir, "eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
    
    return metrics


if __name__ == "__main__":
    image_files = glob.glob("./Abdomen_simulation/cropped_*.png")
    
    with open("./segmentation_centers.json", "r") as f:
        centers_dict = json.load(f)
    
    random.shuffle(image_files)
    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]
    
    save_dir = "./trained_models/navigation"
    
    agent, metrics = train_navigation_agent(
        image_files=train_files,
        centers_dict=centers_dict,
        save_dir=save_dir,
        num_episodes=500
    )
    
    test_env = NavigationEnvironment(test_files, centers_dict)
    eval_metrics = evaluate_agent(test_env, agent, save_dir=os.path.join(save_dir, "evaluation"))
    
    print("Training and evaluation completed!")
