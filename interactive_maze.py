## Recycling previously set up environment
import numpy as np
import matplotlib.pyplot as plt

# 0 = free cell
# 1 = wall
# 2 = hole (terminal, negative reward)
# 3 = goal (terminal, positive reward)

maze = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 2, 0, 0, 1, 0],
    [0, 0, 0, 2, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [2, 0, 1, 0, 0, 3]
])

ROWS, COLS = maze.shape
START = (0, 0)
GOAL  = (5, 5)

# shape: (rows, cols, 4 actions)
Q = np.zeros((ROWS, COLS, 4))

alpha   = 0.5   # learning rate
gamma   = 0.9   # discount factor
epsilon = 0.2   # exploration rate
episodes = 500  

# action directions: 0=up, 1=down, 2=left, 3=right
DIRS = [(-1,0), (1,0), (0,-1), (0,1)]

def step(state, action):
    r, c = state
    dr, dc = DIRS[action]
    nr, nc = r + dr, c + dc

    # if move goes out of bounds or into wall, stay put
    if nr < 0 or nr >= ROWS or nc < 0 or nc >= COLS or maze[nr, nc] == 1:
        nr, nc = r, c

    # assign reward based on what the next cell is
    tile = maze[nr, nc]
    if tile == 3:
        reward = 10       # reached goal
    elif tile == 2:
        reward = -1       # fell in hole
    else:
        reward = -0.04    # normal move — small penalty encourages efficiency

    done = (tile == 3 or tile == 2)
    return (nr, nc), reward, done

def choose_action(state):
    r, c = state
    if np.random.random() < epsilon:
        return np.random.randint(4)           # random action
    else:
        return np.argmax(Q[r, c])             # best known action
    
steps_per_episode = []
deltas_over_time  = []
policy = np.argmax(Q, axis=2)

for episode in range(episodes):
    state = START
    steps = 0

    while True:
        r, c = state

        # select action
        action = choose_action(state)

        # execute action
        next_state, reward, done = step(state, action)
        nr, nc = next_state

        # compute TD error
        delta = reward + gamma * np.max(Q[nr, nc]) - Q[r, c, action]

        # update Q table
        Q[r, c, action] += alpha * delta

        # record
        deltas_over_time.append(delta)
        steps += 1
        state = next_state

        if done or steps > 500:
            break

    steps_per_episode.append(steps)


### Pygame Visualization

import pygame
import numpy as np
import time

pygame.init()

# Grid settings
CELL_SIZE = 80
WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Maze")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (50, 50, 255)

clock = pygame.time.Clock()

# Agent position
agent_pos = list(START)

running = True
auto_move = False

def draw_grid():
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if maze[r, c] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            elif maze[r, c] == 2:
                pygame.draw.rect(screen, RED, rect)
            elif maze[r, c] == 3:
                pygame.draw.rect(screen, GREEN, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)

            pygame.draw.rect(screen, BLACK, rect, 1)

def draw_agent():
    r, c = agent_pos
    center = (c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2)
    pygame.draw.circle(screen, BLUE, center, CELL_SIZE//4)

def move_agent(action):
    global agent_pos
    r, c = agent_pos
    dr, dc = DIRS[action]
    nr, nc = r + dr, c + dc

    if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr, nc] != 1:
        agent_pos = [nr, nc]

while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Press SPACE to start auto-solve
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                auto_move = True

            # Manual controls (arrow keys)
            if event.key == pygame.K_UP:
                move_agent(0)
            if event.key == pygame.K_DOWN:
                move_agent(1)
            if event.key == pygame.K_LEFT:
                move_agent(2)
            if event.key == pygame.K_RIGHT:
                move_agent(3)

    # Auto-follow learned policy
    if auto_move:
        r, c = agent_pos
        action = policy[r, c]
        move_agent(action)
        pygame.time.delay(300)

    draw_grid()
    draw_agent()

    pygame.display.flip()
    clock.tick(10)

pygame.quit()