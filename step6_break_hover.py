import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# 1. Create a wrapper with the flawed "+1 per frame" rule
class HoverHackWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract sideways speed (index 2), up/down speed (index 3), and tilt (index 4)
        sideways_speed = abs(observation[2])
        vertical_speed = abs(observation[3])
        tilt = abs(observation[4])
        
        # Flawed rule: +1 for every frame the rocket is upright and slow
        if tilt < 0.2 and sideways_speed < 0.2 and vertical_speed < 0.2:
            reward += 1.0
            
        return observation, reward, terminated, truncated, info

# 2. Train PPO on the broken reward (using your existing model to speed up learning)
train_env = HoverHackWrapper(gym.make("LunarLander-v3"))
model = PPO.load("ppo_lunar_lander", env=train_env, verbose=1)

print("Training with flawed hover reward...")
model.learn(total_timesteps=300000)
model.save("ppo_lander_broken_hover")
train_env.close()

# 3. Automatically record video clips to catch it hovering
record_env = gym.make("LunarLander-v3", render_mode="rgb_array")
record_env = RecordVideo(
    record_env,
    video_folder="./hover_videos",
    name_prefix="broken_hover_lander",
    episode_trigger=lambda episode_id: True
)

for attempt in range(5):
    observation, info = record_env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = record_env.step(action)
        done = terminated or truncated

record_env.close()
print("Done! Open the './hover_videos' folder to watch the clips.")