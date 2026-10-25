import cv2
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# 1. Reward Wrapper: Forces the rocket to center strictly between the flags
class CenterPadWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_x_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # obs[0] is left/right distance from the pad (0.0 is dead center)
        self.prev_x_dist = abs(obs[0])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_x_dist = abs(obs[0])
        left_leg = obs[6]
        right_leg = obs[7]
        
        # Pay for getting horizontally closer to the center between frames
        x_improvement = (self.prev_x_dist - current_x_dist) * 20.0
        self.prev_x_dist = current_x_dist
        
        # Extra bonus if legs touch down strictly between the flags (|x| < 0.08)
        center_touchdown_bonus = 0.0
        if (left_leg == 1 or right_leg == 1) and current_x_dist < 0.08:
            center_touchdown_bonus = 0.5
            
        # Combine built-in reward, centering improvement, touchdown bonus, and time fee
        shaped_reward = reward + x_improvement + center_touchdown_bonus - 0.05
        return obs, shaped_reward, terminated, truncated, info


# 2. Animation Wrapper: Draws a live telemetry HUD on every frame
class TelemetryHUDWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = np.zeros(8)
        self.last_action = 0
        self.episode_score = 0.0
        self.action_names = {0: "IDLE", 1: "LEFT ENG", 2: "MAIN ENG", 3: "RIGHT ENG"}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_action = 0
        self.episode_score = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        self.last_action = int(action)
        self.episode_score += reward
        return obs, reward, terminated, truncated, info

    def render(self):
        # Grab the raw pixel frame from the game in rgb_array mode
        frame = self.env.render()
        if frame is None:
            return None
            
        frame = frame.copy()
        
        # Draw a dark semi-transparent HUD box in the top-left corner
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (255, 165), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (10, 10), (255, 165), (0, 255, 180), 1)

        # Prepare live telemetry text from the 8 observation numbers
        lines = [
            f"SCORE : {self.episode_score:6.1f}",
            f"THRUST: {self.action_names.get(self.last_action, 'IDLE')}",
            f"X-POS : {self.last_obs[0]:+5.2f} | Y-POS: {self.last_obs[1]:5.2f}",
            f"X-VEL : {self.last_obs[2]:+5.2f} | Y-VEL: {self.last_obs[3]:+5.2f}",
            f"TILT  : {self.last_obs[4]:+5.2f} | SPIN : {self.last_obs[5]:+5.2f}",
            f"LEGS  : L={int(self.last_obs[6])}   | R={int(self.last_obs[7])}"
        ]

        # Render the text onto the video frame
        for i, line in enumerate(lines):
            y_pos = 32 + (i * 22)
            cv2.putText(
                frame, line, (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA
            )

        # Draw a subtle target crosshair line at the center of the landing pad
        cv2.line(frame, (300, 290), (300, 310), (0, 255, 180), 1)
        return frame


# 3. Fine-tune your saved model with the precision centering wrapper
train_env = CenterPadWrapper(gym.make("LunarLander-v3"))
model = PPO.load("ppo_lunar_lander", env=train_env, verbose=1)

print("Fine-tuning for dead-center landings between the flags...")
model.learn(total_timesteps=300000)
model.save("ppo_lander_centered")
train_env.close()

# 4. Record upgraded HUD videos of the centered landings
base_record_env = gym.make("LunarLander-v3", render_mode="rgb_array")
hud_env = TelemetryHUDWrapper(base_record_env)
record_env = RecordVideo(
    hud_env,
    video_folder="./hud_centered_videos",
    name_prefix="hud_bullseye_lander",
    episode_trigger=lambda episode_id: True
)

for attempt in range(5):
    obs, info = record_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = record_env.step(action)
        done = terminated or truncated
    print(f"Recorded Attempt {attempt + 1} | Unwrapped Score: {hud_env.episode_score:.2f}")

record_env.close()
print("Upgraded HUD videos saved in './hud_centered_videos'!")