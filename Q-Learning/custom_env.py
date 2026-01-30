import sys
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame


class RobotTrashCollector(gym.Env):
    """
    Custom Gymnasium environment for a Robot Trash Collector.
    The robot must collect plastic and organic trash and deposit them into the correct bins.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, random_initialization=False):
        super().__init__()
        
        # State Initialization
        self.robot_pos = []
        self.trash1_pos = []
        self.trash2_pos = []
        self.organic_bin_pos = []
        self.plastic_bin_pos = []
        self.carrying_trash_1 = False
        self.carrying_trash_2 = False
        
        # Trash types: 0 = plastic, 1 = Bio
        self.trash1_type = 0 
        self.trash2_type = 1 
        
        self.grid_size = 5
        self.random_initialization = random_initialization

        # Action Space: 0: Up, 1: Down, 2: Left, 3: Right, 4: Pick, 5: Drop
        self.action_space = spaces.Discrete(6)

        # Observation Space: [robot_x, robot_y] (simplified for this implementation)
        # Note: The actual state used in Q-learning might be more complex (see main.py)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([4, 4]),
            dtype=np.int32
        )

        # Rendering
        self.info = {}
        self.done = False
        self.reward = 0
        self.cell_size = 100
        self.window_size = self.grid_size * self.cell_size

        # PyGame Init
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size + 60))
        pygame.display.set_caption("Robot Trash Collector")
        self.clock = pygame.time.Clock()
        
        # Load Assets
        try:
            pygame.mixer.music.load("assets/music/game_music.mp3")
            self.bin_drop = pygame.mixer.Sound("assets/music/bin_drop.mp3")
            self.trash_pick = pygame.mixer.Sound("assets/music/trash_pick.mp3")
            
            self.trash_img_1 = pygame.image.load("assets/plastic_trash.png")
            self.trash_img_2 = pygame.image.load("assets/green_trash.png")
            self.bin_img_1 = pygame.image.load("assets/plastic_bin.png")
            self.bin_img_2 = pygame.image.load("assets/bio_bin.png")
        except Exception as e:
            print(f"Warning: Could not load some assets. {e}")
            # Create placeholders or handle gracefully in a real app
            # For this refactor, we assume assets exist as per original code

        # Scale Images
        self.trash_img_1 = pygame.transform.scale(self.trash_img_1, (self.cell_size, self.cell_size))
        self.trash_img_2 = pygame.transform.scale(self.trash_img_2, (self.cell_size, self.cell_size))
        self.bin_img_1 = pygame.transform.scale(self.bin_img_1, (self.cell_size, self.cell_size))
        self.bin_img_2 = pygame.transform.scale(self.bin_img_2, (self.cell_size, self.cell_size))

        self.trash_img_small_1 = pygame.transform.scale(self.trash_img_1, (self.cell_size // 2, self.cell_size // 2))
        self.trash_img_small_2 = pygame.transform.scale(self.trash_img_2, (self.cell_size // 2, self.cell_size // 2))
        self.bin_img_small_1 = pygame.transform.scale(self.bin_img_1, (self.cell_size // 2, self.cell_size // 2))
        self.bin_img_small_2 = pygame.transform.scale(self.bin_img_2, (self.cell_size // 2, self.cell_size // 2))

        self.font = pygame.font.SysFont('Verdana', 24, bold=True)
        
        self.steps = 0
        self.score = 0
        self.walk_sprites = []

        # Load Animation Sprites
        for i in range(21):
            try:
                path = f"assets/Armature_Walk02_{i:02d}.png"
                sprite = pygame.image.load(path).convert_alpha()
                sprite = pygame.transform.scale(sprite, (self.cell_size, self.cell_size))
                sprite = pygame.transform.flip(sprite, True, False)
                self.walk_sprites.append(sprite)
            except FileNotFoundError:
                pass # Handle missing sprites gracefully

        self.current_walk_frame = 0
        self.last_anim_time = pygame.time.get_ticks()
        self.anim_frame_interval = 80

    def _get_state(self):
        return np.array(self.robot_pos, dtype=np.int32)

    def _get_info(self):
        return self.info

    def _grid_to_pixel(self, pos):
        return pos[1] * self.cell_size, pos[0] * self.cell_size

    def reset(self):
        self.done = False
        if self.random_initialization:
            self.robot_pos = np.array([np.random.choice([0, 1, 2, 3, 4]), np.random.choice([0, 1, 2, 3, 4])])
        else:
            self.robot_pos = np.array([0, 0])

        self.trash1_pos = np.array([1, 1])
        self.trash2_pos = np.array([3, 3])
        self.carrying_trash_1 = False
        self.carrying_trash_2 = False
        self.steps = 0
        self.score = 0
        self.reward = 0
        return self.robot_pos, self._get_info()

    def add_bins(self, bin_positions):
        self.plastic_bin_pos = np.array(bin_positions[0]) if bin_positions[0] is not None else None
        self.organic_bin_pos = np.array(bin_positions[1]) if bin_positions[1] is not None else None

    def step(self, action):
        self.reward = 0.0
        old_robot_pos = self.robot_pos.copy()

        # Update animation frame if moving
        if action in [0, 1, 2, 3] and self.walk_sprites:
            self.current_walk_frame = (self.current_walk_frame + 1) % len(self.walk_sprites)

        # Movement Actions
        if action == 0 and self.robot_pos[0] > 0:
            self.robot_pos[0] -= 1  # Up
        elif action == 1 and self.robot_pos[0] < self.grid_size - 1:
            self.robot_pos[0] += 1  # Down
        elif action == 2 and self.robot_pos[1] > 0:
            self.robot_pos[1] -= 1  # Left
        elif action == 3 and self.robot_pos[1] < self.grid_size - 1:
            self.robot_pos[1] += 1  # Right
        
        # Pick Action
        elif action == 4:
            if np.array_equal(self.robot_pos, self.trash1_pos):
                self.carrying_trash_1 = True
                self.trash1_pos = [-1, -1]
                self.robot_pos[0] += 1
                self.reward = +25
                self.trash_pick.play(0)
            elif np.array_equal(self.robot_pos, self.trash2_pos):
                self.carrying_trash_2 = True
                self.trash2_pos = [-1, -1]
                self.robot_pos[0] += 1
                self.reward = +25
                self.trash_pick.play(0)
            else:
                self.reward = -1

        # Drop Action
        elif action == 5:
            step_reward = 0
            
            # Check drop logic for Trash 1 (Plastic)
            if self.carrying_trash_1:
                if np.array_equal(self.robot_pos, self.plastic_bin_pos):
                    self.trash1_pos = [-2, -2] # Collected
                    self.robot_pos[1] += 1
                    self.carrying_trash_1 = False
                    step_reward += 100
                    self.bin_drop.play(0)
                elif np.array_equal(self.robot_pos, self.organic_bin_pos):
                    step_reward -= 20
                else:
                    step_reward -= 1

            # Check drop logic for Trash 2 (Bio)
            if self.carrying_trash_2:
                if np.array_equal(self.robot_pos, self.organic_bin_pos):
                    self.trash2_pos = [-2, -2] # Collected
                    self.carrying_trash_2 = False
                    step_reward += 100
                    self.bin_drop.play(0)
                elif np.array_equal(self.robot_pos, self.plastic_bin_pos):
                    step_reward -= 20
                else:
                    step_reward -= 1

            if not self.carrying_trash_1 and not self.carrying_trash_2:
                step_reward -= 15

            # Goal check
            if np.array_equal(self.trash1_pos, [-2, -2]) and np.array_equal(self.trash2_pos, [-2, -2]):
                self.done = True
                step_reward += 100

            self.reward = step_reward

        self.steps += 1
        
        # Wall collision penalty
        if action in [0, 1, 2, 3] and np.array_equal(self.robot_pos, old_robot_pos):
            self.reward -= 30
            
        self.reward -= 0.5  # Step penalty

        return self.robot_pos, self.reward, self.done, self._get_info()

    def render(self):
        for event in pygame.event.get():
            if event == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 240, 200))

        # Draw Grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (230, 230, 230), rect, border_radius=12)
                pygame.draw.rect(self.screen, (180, 180, 180), rect, 2, border_radius=12)

        self.robot_pixel_pos = self._grid_to_pixel(self.robot_pos)
        self.update_walk_animation()
        
        if self.walk_sprites:
            robot_sprite = self.walk_sprites[self.current_walk_frame]
        else:
            # Fallback if no sprites loaded
            robot_sprite = pygame.Surface((self.cell_size, self.cell_size))
            robot_sprite.fill((255,0,0))

        shadow = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 70), (10, self.cell_size - 20, self.cell_size - 20, 10))

        # Determine highlight and draw robot
        highlight_color = None
        
        if np.array_equal(self.robot_pos, self.trash1_pos):
            highlight_color = (255, 255, 100, 120)
        elif np.array_equal(self.robot_pos, self.trash2_pos):
            highlight_color = (255, 255, 100, 120)
        elif np.array_equal(self.robot_pos, self.plastic_bin_pos):
            highlight_color = (100, 100, 255, 120)
        elif np.array_equal(self.robot_pos, self.organic_bin_pos):
            highlight_color = (100, 255, 100, 120)

        # Draw Shadow & Robot
        self.screen.blit(shadow, self.robot_pixel_pos)
        self.screen.blit(robot_sprite, self.robot_pixel_pos)

        # Draw Holds (Small items if carrying)
        if self.carrying_trash_1: 
             self.screen.blit(self.trash_img_small_1, (self.robot_pixel_pos[0] + self.cell_size // 2, self.robot_pixel_pos[1] + self.cell_size // 2))
        
        if self.carrying_trash_2:
             self.screen.blit(self.trash_img_small_2, (self.robot_pixel_pos[0] + self.cell_size // 2, self.robot_pixel_pos[1] + self.cell_size // 2))

        # Basic Object Drawing Helper
        def draw_object(pos, img, small_img=None, is_held=False):
            if np.array_equal(pos, [-1, -1]) or np.array_equal(pos, [-2, -2]):
                return # Held or Collected
            self.screen.blit(img, self._grid_to_pixel(pos))

        draw_object(self.trash1_pos, self.trash_img_1)
        draw_object(self.trash2_pos, self.trash_img_2)
        draw_object(self.plastic_bin_pos, self.bin_img_1)
        draw_object(self.organic_bin_pos, self.bin_img_2)

        if highlight_color:
            highlight = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            highlight.fill(highlight_color)
            self.screen.blit(highlight, self.robot_pixel_pos)

        # Status Bar
        status_bg = pygame.Surface((self.window_size, 60))
        status_bg.fill((0, 0, 0))
        self.screen.blit(status_bg, (0, self.window_size))
        status_text = self.font.render(f"steps: {self.steps}     Reward: {round(self.reward, 2)}", True, (255, 255, 255))
        self.screen.blit(status_text, (20, self.window_size + 15))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def update_walk_animation(self):
        now = pygame.time.get_ticks()
        if now - self.last_anim_time >= self.anim_frame_interval:
            if self.walk_sprites:
                self.current_walk_frame = (self.current_walk_frame + 1) % len(self.walk_sprites)
            self.last_anim_time = now

    def show_game_over_banner(self):
        banner = pygame.Surface((self.window_size, 80))
        banner.set_alpha(180)
        banner.fill((0, 0, 0))
        self.screen.blit(banner, (0, self.window_size // 2 - 40))
        text = self.font.render("All Trash Collected!", True, (255, 255, 255))
        self.screen.blit(text, (self.window_size // 2 - text.get_width() // 2, self.window_size // 2 - 10))
        pygame.display.flip()
        pygame.time.delay(2000)

    def close(self):
        pygame.quit()


def create_env(bin_positions=None, random_initialization=False):
    """Factory function to create the environment."""
    env = RobotTrashCollector(random_initialization=random_initialization)
    env.add_bins(bin_positions)
    return env


def run_agent_demo(env, q_table1_path, q_table2_path):
    """Runs a demonstration of the trained agent."""
    try:
        q_table1 = np.load(q_table1_path)
        q_table2 = np.load(q_table2_path)
    except FileNotFoundError:
        print("Error: Q-table files not found.")
        return

    print("Q-tables loaded successfully.")
    phase = 1
    done = False
    
    state, _ = env.reset()
    total_reward = 0
    # Map state to Q-table index: x, y, carry1, carry2
    state = (state[0], state[1], int(env.carrying_trash_1), int(env.carrying_trash_2))
    
    current_q_table = q_table1
    
    import time
    
    while not done:
        env.render()
        time.sleep(0.5)
        
        # Choose action (Exploit)
        action = np.argmax(current_q_table[state])
        
        next_pos, reward, done, _ = env.step(action)
        next_state = (next_pos[0], next_pos[1], int(env.carrying_trash_1), int(env.carrying_trash_2))
        
        state = next_state
        total_reward += reward

        # Simple phase switch logic based on trash 1 collection
        if phase == 1 and np.array_equal(env.trash1_pos, [-2, -2]):
            current_q_table = q_table2
    
    print("Agent finished running.")
    print(f"Final Score: {total_reward}")
    env.show_game_over_banner()
    env.close()