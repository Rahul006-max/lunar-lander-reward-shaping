import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# 1. Create the environment in "rgb_array" mode for video recording
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# 2. Wrap the environment to record videos of the attempts
env = RecordVideo(
    env, 
    video_folder="./landing_videos", 
    name_prefix="trained_lander",
    episode_trigger=lambda episode_id: True
)

# 3. Load the saved model
model = PPO.load("ppo_lunar_lander")

# 4. Run 5 attempts and check the scores
for attempt in range(5):
    observation, info = env.reset()
    done = False
    total_score = 0
    
    while not done:
        # Ask the trained AI to pick the best action based on the 8 numbers
        action, _states = model.predict(observation, deterministic=True)
        
        # Pass the chosen action into the game
        observation, reward, terminated, truncated, info = env.step(action)
        total_score += reward
        
        # Check if the rocket landed safely or crashed
        done = terminated or truncated
        
    print(f"Attempt {attempt + 1} Score: {total_score:.2f}")

env.close()
print("Videos saved in the './landing_videos' folder!")