import pygame
import gym
from stable_baselines3 import DQN, PPO
import numpy as np
import random

# Define the Snake environment (same as your initial SnakeEnv)
class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, grid_size=10, render_mode=None):
        super(SnakeEnv, self).__init__()

        self.grid_size = grid_size
        self.window_size = grid_size * 20
        self.render_mode = render_mode

        # Initialize Pygame if human rendering is required
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        # Action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: [head_x, head_y, food_x, food_y]
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(4,),  # Changed from 8 to 4
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize snake in the middle
        self.snake = [(self.grid_size//2, self.grid_size//2)]
        self.direction = 1  # Start moving right
        self.food = self._place_food()
        self.steps = 0
        self.score = 0

        return self._get_obs(), {}

    def _place_food(self):
        while True:
            food = (random.randint(0, self.grid_size-1),
                   random.randint(0, self.grid_size-1))
            if food not in self.snake:
                return food

    def _get_obs(self):
        head = self.snake[0]
        return np.array([
            head[0] / self.grid_size,  # head x position
            head[1] / self.grid_size,  # head y position
            self.food[0] / self.grid_size,  # food x position
            self.food[1] / self.grid_size,  # food y position
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = 0
        done = False

        # Update direction based on action
        self.direction = action

        # Get new head position
        head = self.snake[0]
        if action == 0:  # UP
            new_head = (head[0], head[1] - 1)
        elif action == 1:  # RIGHT
            new_head = (head[0] + 1, head[1])
        elif action == 2:  # DOWN
            new_head = (head[0], head[1] + 1)
        else:  # LEFT
            new_head = (head[0] - 1, head[1])

        # Check for collision with walls or self
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            return self._get_obs(), -10, True, False, {}

        # Move snake
        self.snake.insert(0, new_head)

        # Check if food is eaten
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()
            # Small reward/penalty based on distance to food
            prev_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 0.1 if new_dist < prev_dist else -0.1

        # End episode if it's taking too long
        if self.steps >= 100:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self):
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))  # Clear screen
            for x, y in self.snake:
                pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(x * 20, y * 20, 20, 20))
            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.food[0] * 20, self.food[1] * 20, 20, 20))
            pygame.display.flip()

            self.clock.tick(self.metadata['render_fps'])

# Load the trained models
dqn_model = DQN.load("snake_dqn_model")
ppo_model = PPO.load("snake_ppo_model")

# Initialize the environment
env = SnakeEnv(render_mode="human")

# Game loop for running the snake with AI-controlled actions
done = False
obs, _ = env.reset()

while not done:
    # Get the action from the model
    action, _states = dqn_model.predict(obs)  # Replace dqn_model with ppo_model if you want to test PPO
    
    # Take the action in the environment
    obs, reward, done, _, _ = env.step(action)
    
    # Render the game (human view)
    env.render()

env.close()
