import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment (no render_mode this time so it trains much faster)
env = gym.make("LunarLander-v3")

# Initialize PPO with the MlpPolicy network
model = PPO("MlpPolicy", env, verbose=1)

print("Starting training...")
# Train the model for 1 million steps 
model.learn(total_timesteps=1000000)

# Save the model when it finishes
model.save("ppo_lunar_lander")
print("Model saved as ppo_lunar_lander.zip!")

env.close()