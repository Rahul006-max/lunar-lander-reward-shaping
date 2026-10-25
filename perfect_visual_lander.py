import cv2
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# 1. Potential-Based Precision Wrapper (No per-frame farming loopholes)
class PerfectCenterWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_potential = None

    def _calc_potential(self, obs):
        # obs[0] = left/right of pad, obs[2] = sideways speed, obs[4] = tilt
        x_dist = abs(obs[0])
        x_speed = abs(obs[2])
        tilt = abs(obs[4])
        # Higher (closer to 0) is better
        return -(40.0 * x_dist + 20.0 * x_speed + 40.0 * tilt)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_potential = self._calc_potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Pay strictly for the change between frames so hovering/sliding pays 0
        current_potential = self._calc_potential(obs)
        shaping_bonus = current_potential - self.prev_potential
        self.prev_potential = current_potential
        
        # Extra one-time bonus on the final frame if it lands dead-center
        if terminated and reward >= 100:
            center_accuracy = max(0.0, 1.0 - (abs(obs[0]) / 0.15))
            shaping_bonus += 50.0 * center_accuracy

        # Subtract a small fee per frame so waiting costs something
        shaped_reward = reward + shaping_bonus - 0.05
        return obs, shaped_reward, terminated, truncated, info


# 2. Visual Environment Redesign Wrapper (Space Sky, Lunar Terrain, Cockpit HUD)
class SpaceThemeVisualWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = np.zeros(8)
        self.last_action = 0
        self.episode_score = 0.0
        self.frame_count = 0
        
        # Pre-generate a fixed starfield and space gradient (400x600)
        rng = np.random.default_rng(42)
        self.star_coords = rng.integers(0, [320, 600], size=(120, 2))
        self.star_Phase = rng.uniform(0, 6.28, size=120)
        
        # Create a vertical deep-space gradient (dark navy at top to purple-indigo near horizon)
        y_grad = np.linspace(0, 1, 400)[:, None]
        self.space_bg = np.zeros((400, 600, 3), dtype=np.uint8)
        self.space_bg[:, :, 0] = (10 + 25 * y_grad).astype(np.uint8)  # Red
        self.space_bg[:, :, 1] = (12 + 18 * y_grad).astype(np.uint8)  # Green
        self.space_bg[:, :, 2] = (28 + 45 * y_grad).astype(np.uint8)  # Blue

        # Draw a distant shaded planet/moon in the top-right sky
        cv2.circle(self.space_bg, (500, 85), 38, (45, 70, 110), -1, cv2.LINE_AA)
        cv2.circle(self.space_bg, (490, 78), 36, (18, 22, 42), -1, cv2.LINE_AA)

        # Pre-generate a lunar regolith gradient for the terrain
        self.lunar_ground = np.zeros((400, 600, 3), dtype=np.uint8)
        self.lunar_ground[:, :, 0] = (75 - 35 * y_grad).astype(np.uint8)
        self.lunar_ground[:, :, 1] = (80 - 35 * y_grad).astype(np.uint8)
        self.lunar_ground[:, :, 2] = (95 - 35 * y_grad).astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_action = 0
        self.episode_score = 0.0
        self.frame_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        self.last_action = int(action)
        self.episode_score += reward
        self.frame_count += 1
        return obs, reward, terminated, truncated, info

    def render(self):
        raw_frame = self.env.render()
        if raw_frame is None:
            return None

        frame = raw_frame.copy()

        # A. Replace black sky pixels with deep-space background + twinkling stars
        sky_mask = (frame[:, :, 0] < 15) & (frame[:, :, 1] < 15) & (frame[:, :, 2] < 15)
        bg = self.space_bg.copy()
        for (sy, sx), phase in zip(self.star_coords, self.star_Phase):
            brightness = int(160 + 95 * np.sin(self.frame_count * 0.15 + phase))
            bg[sy, sx] = (brightness, brightness, min(255, brightness + 30))
        frame[sky_mask] = bg[sky_mask]

        # B. Replace flat white terrain pixels with shaded lunar regolith
        ground_mask = (frame[:, :, 0] > 235) & (frame[:, :, 1] > 235) & (frame[:, :, 2] > 235)
        frame[ground_mask] = self.lunar_ground[ground_mask]

        # C. Draw glowing landing pad target zone at center (X = 300)
        aligned = abs(self.last_obs[0]) < 0.08
        pad_color = (0, 255, 160) if aligned else (0, 190, 255)
        cv2.line(frame, (265, 300), (335, 300), pad_color, 2, cv2.LINE_AA)
        cv2.circle(frame, (300, 300), 4, (0, 255, 120), -1, cv2.LINE_AA)

        # D. Draw sleek dark-glass Cockpit HUD in top-left
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (275, 178), (12, 16, 28), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (12, 12), (275, 178), (60, 180, 255), 1, cv2.LINE_AA)

        # Header & Score
        cv2.putText(frame, "LUNAR GUIDANCE TELEMETRY", (22, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 200, 255), 1, cv2.LINE_AA)
        score_color = (80, 255, 150) if self.episode_score >= 200 else (230, 230, 230)
        cv2.putText(frame, f"SCORE: {self.episode_score:6.1f}", (22, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, score_color, 1, cv2.LINE_AA)

        # Telemetry metrics (Observations 1 to 6)
        metrics = [
            f"X-OFFSET : {self.last_obs[0]:+5.2f}  | ALT : {self.last_obs[1]:5.2f}",
            f"VX-SPEED : {self.last_obs[2]:+5.2f}  | VY  : {self.last_obs[3]:+5.2f}",
            f"TILT RAD : {self.last_obs[4]:+5.2f}  | SPIN: {self.last_obs[5]:+5.2f}",
        ]
        for idx, text in enumerate(metrics):
            cv2.putText(frame, text, (22, 78 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 225, 240), 1, cv2.LINE_AA)

        # Thruster Indicator Pills (Actions 1, 2, 3)
        thrusters = [(1, "LEFT", 22), (2, "MAIN", 88), (3, "RIGHT", 154)]
        for act_id, label, x_pos in thrusters:
            is_on = (self.last_action == act_id)
            box_col = (255, 140, 40) if is_on else (35, 45, 65)
            txt_col = (15, 15, 15) if is_on else (120, 135, 155)
            cv2.rectangle(frame, (x_pos, 132), (x_pos + 56, 150), box_col, -1)
            cv2.putText(frame, label, (x_pos + 8, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, txt_col, 1, cv2.LINE_AA)

        # Leg Contact Status (Observations 7 and 8)
        l_leg, r_leg = int(self.last_obs[6]), int(self.last_obs[7])
        if l_leg and r_leg:
            status_txt, status_col = "STATUS: TOUCHDOWN LOCKED", (50, 255, 130)
        elif l_leg or r_leg:
            status_txt, status_col = "STATUS: PARTIAL CONTACT", (255, 210, 50)
        else:
            status_txt, status_col = "STATUS: DESCENT PHASE", (150, 190, 255)

        cv2.putText(frame, status_txt, (22, 168),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_col, 1, cv2.LINE_AA)

        return frame


# 3. Fine-tune with a smaller learning rate so PPO updates remain stable
train_env = PerfectCenterWrapper(gym.make("LunarLander-v3"))
model = PPO.load("ppo_lunar_lander", env=train_env, learning_rate=1e-4, verbose=1)

print("Refining model for consistent 250+ center landings...")
model.learn(total_timesteps=400000)
model.save("ppo_lander_perfect")
train_env.close()

# 4. Record 5 attempts with the redesigned Space Theme environment
base_env = gym.make("LunarLander-v3", render_mode="rgb_array")
themed_env = SpaceThemeVisualWrapper(base_env)
record_env = RecordVideo(
    themed_env,
    video_folder="./space_theme_videos",
    name_prefix="space_lander",
    episode_trigger=lambda episode_id: True
)

scores = []
for attempt in range(5):
    obs, info = record_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = record_env.step(action)
        done = terminated or truncated
    scores.append(themed_env.episode_score)
    print(f"Attempt {attempt + 1} | Unwrapped Score: {themed_env.episode_score:.2f}")

record_env.close()
print(f"\nAverage Unwrapped Score: {sum(scores)/len(scores):.2f}")
print("Redesigned videos saved in './space_theme_videos'!")