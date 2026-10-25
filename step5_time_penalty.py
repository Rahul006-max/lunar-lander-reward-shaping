import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create a plain Wrapper to modify the reward
class TimePenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Step the game forward and get the built-in outputs
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Subtract a tiny charge per frame so waiting costs something
        modified_reward = reward - 0.05
        
        # Pass the modified reward total on to PPO
        return observation, modified_reward, terminated, truncated, info

# 2. Create the environment and wrap it
base_env = gym.make("LunarLander-v3")
wrapped_env = TimePenaltyWrapper(base_env)

# 3. Load your already-trained model from Step 3 instead of starting randomly
model = PPO.load("ppo_lunar_lander", env=wrapped_env, verbose=1)

print("Fine-tuning with a per-frame time penalty (faster)...")
# Train for 250,000 steps instead of 1,000,000 (cuts wait time by 75%)
model.learn(total_timesteps=250000)

model.save("ppo_lander_time_penalty")
wrapped_env.close()

# 4. Test the trained model with the wrapper OFF over 10 attempts
eval_env = gym.make("LunarLander-v3")
scores = []

for attempt in range(10):
    observation, info = eval_env.reset()
    done = False
    episode_score = 0
    
    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = eval_env.step(action)
        episode_score += reward
        done = terminated or truncated
        
    scores.append(episode_score)
    print(f"Attempt {attempt + 1} Score: {episode_score:.2f}")

eval_env.close()
print(f"\nAverage Score over 10 attempts: {sum(scores) / len(scores):.2f}")