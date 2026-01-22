import gym
import numpy as np
import os
from pathlib import Path
import time
import sys

# --- FIX: IMPORT MINIGRID ---
# We try to import the library to register the environments.
try:
    import gym_minigrid
except ImportError:
    try:
        import minigrid
    except ImportError:
        print("Error: MiniGrid not installed.")
        print("Please run: pip install gym-minigrid")
        sys.exit(1)

# --- CONFIG ---
ENV_NAME = 'MiniGrid-Empty-5x5-v0' 
TOTAL_EPISODES = 1000
MAX_STEPS = 100
DATA_PATH = Path(__file__).parent.parent / "data" / "episodes"
DATA_PATH.mkdir(parents=True, exist_ok=True)

def main():
    print(f"Starting Data Collection: {ENV_NAME}")
    
    # Try initializing with modern render_mode syntax
    try:
        env = gym.make(ENV_NAME, render_mode='rgb_array')
    except:
        # Fallback for older Gym versions
        env = gym.make(ENV_NAME)

    # We need to ensure we can capture images
    env.reset()
    
    total_frames = 0
    
    print(f"Collecting {TOTAL_EPISODES} episodes...")
    
    for ep in range(TOTAL_EPISODES):
        obs = env.reset()
        frames = []
        actions = []
        
        for t in range(MAX_STEPS):
            # 1. Render Frame
            try:
                # Modern Gym
                img = env.render()
                if img is None: # Fallback if render_mode wasn't accepted
                     img = env.render(mode='rgb_array')
            except:
                # Legacy Gym
                img = env.render(mode='rgb_array')
            
            # Validation: Ensure we got an image
            if img is None or not isinstance(img, np.ndarray):
                print("⚠️ Warning: Could not render image. Check gym/minigrid version.")
                break

            # 2. Resize to 64x64 for VQ-VAE (MiniGrid is usually very small)
            import cv2
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
            
            frames.append(img)
            
            # 3. Random Action (With bias towards movement)
            # MiniGrid actions: 0:done, 1:left, 2:right, 3:forward, 4:pickup, 5:drop, 6:toggle
            # We filter out 'done' (0) to keep the episode going
            if np.random.rand() < 0.8: 
                # 80% chance: Move or Turn (Actions 1, 2, 3)
                action = np.random.choice([1, 2, 3])
            else:
                # 20% chance: Random other action
                action = env.action_space.sample()
            
            # Execute
            try:
                # Modern Gym returns 5 values
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated: break
            except ValueError:
                # Old Gym returns 4 values
                obs, reward, done, info = env.step(action)
                if done: break
            
            actions.append(action)
            
        # Save Episode if it has frames
        if len(frames) > 0:
            frames_np = np.array(frames) # (T, 64, 64, 3)
            actions_np = np.array(actions)
            
            save_path = DATA_PATH / f"episode_{ep:04d}.npz"
            np.savez_compressed(save_path, frames=frames_np, action=actions_np)
            
            total_frames += len(frames)
            if ep % 10 == 0:
                print(f"Saved episode {ep}/{TOTAL_EPISODES} (Total frames: {total_frames})")

    print("Data Collection Complete.")

if __name__ == "__main__":
    main()