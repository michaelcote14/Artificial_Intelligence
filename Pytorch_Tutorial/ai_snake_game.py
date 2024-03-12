import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 1400


class SnakeGame_AI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.snake_length = 1
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # This is the snake moving one block at a time
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        neural_reward = 0
        game_over = False
        # This checks if the snake has hit itself or the wall or if it has taken too long to get to the food
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            neural_reward = -10
            return neural_reward, game_over, self.score

        # 4. If the snake hits the food, it will grow and the score will increase
        # and a new food will be placed
        if self.head == self.food:
            self.score += 1
            self.snake_length += 1
            neural_reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return neural_reward, game_over, self.score

    def is_collision(self, point=None):
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left] # these are the turns the snake can make

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)

        # This is what decides which direction the snake will go
        if np.array_equal(action, [1, 0, 0]): # Keep going straight
            new_direction = clock_wise[index]
        elif np.array_equal(action, [0, 1, 0]): # Turn right
            next_index = (index + 1) % 4
            new_direction = clock_wise[next_index]
        else: # Turn left
            next_index = (index - 1) % 4
            new_direction = clock_wise[next_index]

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
