#!/usr/bin/env python3
"""
Generate synthetic test data for local RPCA testing.
Creates small fake datasets that mimic the racing game structure.
"""

import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import h5py

from src.data import Dataset, Episode
from src.data.segment import SegmentId


def create_synthetic_racing_frames(num_frames: int, height: int = 48, width: int = 64, 
                                 channels: int = 3) -> np.ndarray:
    """
    Create synthetic racing game frames with realistic structure:
    - Global motion (road, background) -> Low-rank component
    - Sparse events (crashes, overtakes) -> Sparse component
    """
    frames = np.zeros((num_frames, channels, height, width), dtype=np.float32)
    
    # Create road pattern (low-rank structure)
    road_pattern = np.zeros((height, width))
    
    # Road lines
    road_pattern[height//2-2:height//2+2, :] = 0.8  # Center line
    road_pattern[:, width//4] = 0.6      # Left line  
    road_pattern[:, 3*width//4] = 0.6    # Right line
    
    # Add perspective effect
    for y in range(height):
        perspective_factor = (y / height) ** 0.5
        road_pattern[y, :] *= perspective_factor
        
    # Background gradient
    background = np.linspace(0.3, 0.7, height).reshape(-1, 1)
    background = np.broadcast_to(background, (height, width))
    
    for t in range(num_frames):
        # Animate road (low-rank temporal structure)
        road_shift = (t * 2) % height
        shifted_road = np.roll(road_pattern, road_shift, axis=0)
        
        # Animate background
        bg_shift = (t * 1) % height  
        shifted_bg = np.roll(background, bg_shift, axis=0)
        
        base_frame = shifted_bg + shifted_road
        
        # Add cars (moving objects with some low-rank structure)
        car1_x = int(width * 0.3 + 0.1 * width * np.sin(t * 0.1))
        car1_y = int(height * 0.7 + 0.05 * height * np.cos(t * 0.15))
        
        car2_x = int(width * 0.7 + 0.08 * width * np.sin(t * 0.12 + 1))
        car2_y = int(height * 0.4 + 0.1 * height * np.cos(t * 0.1))
        
        # Draw simple cars (3x3 squares)
        if 1 <= car1_x < width-1 and 1 <= car1_y < height-1:
            base_frame[car1_y-1:car1_y+2, car1_x-1:car1_x+2] = 1.0
            
        if 1 <= car2_x < width-1 and 1 <= car2_y < height-1:
            base_frame[car2_y-1:car2_y+2, car2_x-1:car2_x+2] = 0.9
            
        # Add sparse events (crashes, explosions) 
        if t % 15 == 0 and t > 0:  # Sparse crash every 15 frames
            crash_x = np.random.randint(5, width-5)
            crash_y = np.random.randint(5, height-5)
            crash_size = 8
            
            # Create explosion pattern (sparse component)
            explosion = np.random.random((crash_size, crash_size)) * 3.0
            explosion *= (explosion > 0.7)  # Make it sparse
            
            y1, y2 = max(0, crash_y-crash_size//2), min(height, crash_y+crash_size//2)
            x1, x2 = max(0, crash_x-crash_size//2), min(width, crash_x+crash_size//2)
            
            ey1, ey2 = max(0, crash_size//2-crash_y), explosion.shape[0] - max(0, crash_y+crash_size//2-height)
            ex1, ex2 = max(0, crash_size//2-crash_x), explosion.shape[1] - max(0, crash_x+crash_size//2-width)
            
            base_frame[y1:y2, x1:x2] += explosion[ey1:ey2, ex1:ex2]
            
        # Replicate across channels with slight variations
        for c in range(channels):
            channel_variation = 0.1 * np.random.random((height, width)) 
            frames[t, c] = np.clip(base_frame + channel_variation, 0, 1)
            
    # Convert to [-1, 1] range (matching the real data preprocessing)
    frames = frames * 2 - 1
    
    return frames


def create_synthetic_actions(num_frames: int, num_actions: int = 66) -> np.ndarray:
    """Create synthetic action sequences."""
    actions = np.zeros(num_frames, dtype=np.int32)
    
    # Create semi-realistic action patterns
    for t in range(num_frames):
        if t % 20 < 10:  # Go straight/accelerate
            actions[t] = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])  
        else:  # Turn/brake
            actions[t] = np.random.choice([3, 4, 5, 6], p=[0.3, 0.3, 0.2, 0.2])
            
    return actions


def create_synthetic_episode(episode_length: int = 100) -> Episode:
    """Create a complete synthetic episode."""
    frames = create_synthetic_racing_frames(episode_length)
    actions = create_synthetic_actions(episode_length) 
    
    # Convert to torch tensors
    obs = torch.from_numpy(frames).float()
    act = torch.from_numpy(actions).long()
    
    # Create rewards (simple: +1 for forward progress, -10 for crashes)
    rew = torch.ones(episode_length) * 0.1
    for t in range(episode_length):
        if t % 15 == 0 and t > 0:  # Crash frames (same as sparse events)
            rew[t] = -10.0
        elif t % 5 == 0:  # Good progress
            rew[t] = 1.0
            
    # Episode end and truncation
    end = torch.zeros(episode_length, dtype=torch.uint8)
    end[-1] = 1  # Episode ends at last frame
    
    trunc = torch.zeros(episode_length, dtype=torch.uint8)
    
    info = {"synthetic": True, "episode_type": "racing"}
    
    return Episode(obs, act, rew, end, trunc, info)


def create_test_dataset(output_dir: Path, num_train_episodes: int = 10, 
                       num_test_episodes: int = 3, episode_length: int = 50):
    """Create complete test dataset structure."""
    
    output_dir = Path(output_dir)
    
    # Create directories
    low_res_dir = output_dir / "low_res"
    full_res_dir = output_dir / "full_res"
    
    train_dir = low_res_dir / "train" 
    test_dir = low_res_dir / "test"
    
    for dir_path in [train_dir, test_dir, full_res_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        
    print(f"Creating synthetic dataset in {output_dir}")
    print(f"Episodes: {num_train_episodes} train, {num_test_episodes} test")
    print(f"Episode length: {episode_length} frames")
    
    # Create datasets
    train_dataset = Dataset(train_dir, None, "train_dataset", save_on_disk=True)
    test_dataset = Dataset(test_dir, None, "test_dataset", save_on_disk=True)
    
    # Generate training episodes
    print("Generating training episodes...")
    for i in tqdm(range(num_train_episodes)):
        episode = create_synthetic_episode(episode_length)
        train_dataset.add_episode(episode)
        
    # Generate test episodes  
    print("Generating test episodes...")
    for i in tqdm(range(num_test_episodes)):
        episode = create_synthetic_episode(episode_length)
        test_dataset.add_episode(episode)
        
    # Save datasets
    train_dataset.save_to_default_path()
    test_dataset.save_to_default_path()
    
    # Create fake full-res data (just upscaled versions)
    print("Creating full-resolution data...")
    full_res_files = {}
    
    for split, dataset in [("train", train_dataset), ("test", test_dataset)]:
        for episode_id in range(dataset.num_episodes):
            episode = dataset.load_episode(episode_id)
            
            # Create fake full-res by upscaling
            B, C, H, W = episode.obs.shape
            full_res_frames = torch.nn.functional.interpolate(
                episode.obs, size=(H*4, W*4), mode='bicubic', align_corners=False
            )
            
            # Save as HDF5 (matching the real data format)
            filename = f"synthetic_{split}_{episode_id:03d}.hdf5"
            filepath = full_res_dir / filename
            
            with h5py.File(filepath, 'w') as f:
                for t in range(len(full_res_frames)):
                    frame = full_res_frames[t].permute(1, 2, 0).numpy()  # CHW -> HWC
                    frame = ((frame + 1) * 127.5).astype(np.uint8)  # [-1,1] -> [0,255]
                    
                    f[f'frame_{t}_x'] = frame
                    f[f'frame_{t}_y'] = episode.act[t].numpy()
                    
            full_res_files[f"{split}/{filename}"] = filepath
            
    print(f"\n✅ Test dataset created successfully!")
    print(f"📁 Location: {output_dir}")
    print(f"📊 Stats:")
    print(f"   Training episodes: {train_dataset.num_episodes} ({train_dataset.num_steps} steps)")
    print(f"   Test episodes: {test_dataset.num_episodes} ({test_dataset.num_steps} steps)")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Create synthetic test data for RPCA testing')
    parser.add_argument('--output', type=str, default='./test_data',
                       help='Output directory for test data')
    parser.add_argument('--train-episodes', type=int, default=5,
                       help='Number of training episodes')
    parser.add_argument('--test-episodes', type=int, default=2, 
                       help='Number of test episodes')
    parser.add_argument('--episode-length', type=int, default=30,
                       help='Length of each episode (frames)')
    parser.add_argument('--tiny', action='store_true',
                       help='Create extra small dataset for quick testing')
    
    args = parser.parse_args()
    
    if args.tiny:
        args.train_episodes = 2
        args.test_episodes = 1  
        args.episode_length = 10
        print("🔬 Creating TINY dataset for ultra-fast testing")
        
    output_path = create_test_dataset(
        args.output,
        num_train_episodes=args.train_episodes,
        num_test_episodes=args.test_episodes, 
        episode_length=args.episode_length
    )
    
    print(f"\n🚀 Ready to test! Run:")
    print(f"python run_rpca_experiments.py --data {output_path} --experiments baseline rpca_default --quick")


if __name__ == "__main__":
    main()