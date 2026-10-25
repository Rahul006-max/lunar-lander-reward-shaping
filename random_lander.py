import gymnasium as gym

# Create the environment by its ID with the game window on
env = gym.make("LunarLander-v3", render_mode="human")

# Reset to start the first attempt
observation, info = env.reset()

for _ in range(1000):
    # Pick a random action (0: do nothing, 1: left engine, 2: main engine, 3: right engine)
    action = env.action_space.sample()
    
    # Step to act: returns the new observation, the reward, and if the attempt ended
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Print the eight numbers the AI sees (coordinates, speed, tilt, legs)
    print(f"Observation: {observation}")
    
    # If the rocket crashes or lands, restart
    if terminated or truncated:
        print("Attempt ended. Restarting...")
        observation, info = env.reset()

env.close()