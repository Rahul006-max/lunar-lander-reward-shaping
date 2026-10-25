import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# 1. Create a wrapper that rewards frame-to-frame improvement + charges a time fee
class FixedShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_stability = None

    def _calc_stability(self, obs):
        # Measure how slow and upright the rocket is (closer to 0 is better)
        sideways_speed = abs(obs[2])
        vertical_speed = abs(obs[3])
        tilt = abs(obs[4])
        return -(sideways_speed + vertical_speed + tilt)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # Store the initial stability score on the first frame of the attempt
        self.prev_stability = self._calc_stability(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate current stability and how much it improved since the last frame
        current_stability = self._calc_stability(observation)
        improvement = current_stability - self.prev_stability
        self.prev_stability = current_stability
        
        # Pay for getting better between frames, minus a small fee for waiting
        shaped_reward = reward + (improvement * 10.0) - 0.05
        
        return observation, shaped_reward, terminated, truncated, info

# 2. Train PPO with the fixed reward (faster fine-tuning from base model)
train_env = FixedShapingWrapper(gym.make("LunarLander-v3"))
model = PPO.load("ppo_lunar_lander", env=train_env, verbose=1)

print("Training with fixed reward shaping...")
model.learn(total_timesteps=250000)
model.save("ppo_lander_fixed")
train_env.close()

# 3. Record the fixed landing clips
record_env = gym.make("LunarLander-v3", render_mode="rgb_array")
record_env = RecordVideo(
    record_env,
    video_folder="./fixed_videos",
    name_prefix="fixed_lander",
    episode_trigger=lambda episode_id: True
)

for attempt in range(3):
    observation, info = record_env.reset()
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = record_env.step(action)
        done = terminated or truncated

record_env.close()

# 4. Final Verification: Test with the wrapper OFF over 10 attempts
eval_env = gym.make("LunarLander-v3")
scores = []

for attempt in range(10):
    observation, info = eval_env.reset()
    done = False
    episode_score = 0
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = eval_env.step(action)
        episode_score += reward
        done = terminated or truncated
    scores.append(episode_score)
    print(f"Unwrapped Attempt {attempt + 1} Score: {episode_score:.2f}")

eval_env.close()
print(f"\nFinal Average Score (10 attempts, wrapper OFF): {sum(scores) / len(scores):.2f}")