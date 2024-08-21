import pygame
import neat
import os
import random
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH = 800
HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Snake (CPU Version)")

class Snake:
    def __init__(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
        self.score = 0
        self.fitness = 0
        self.steps = 0
        self.steps_without_food = 0

    def move(self):
        head = self.body[0]
        new_head = ((head[0] + self.direction[0]) % GRID_WIDTH,
                    (head[1] + self.direction[1]) % GRID_HEIGHT)
        self.body.insert(0, new_head)
        self.steps += 1
        self.steps_without_food += 1

    def grow(self):
        self.score += 1
        self.steps_without_food = 0

    def check_collision(self):
        return len(self.body) != len(set(self.body))

class Food:
    def __init__(self):
        self.position = self.generate_position()

    def generate_position(self):
        return (random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1))

class Obstacle:
    def __init__(self):
        self.positions = self.generate_positions()

    def generate_positions(self):
        num_obstacles = random.randint(5, 10)
        return [(random.randint(0, GRID_WIDTH - 1),
                 random.randint(0, GRID_HEIGHT - 1))
                for _ in range(num_obstacles)]

def draw_game(screen, snake, food, obstacles, generation, genome_id):
    screen.fill(BLACK)

    # Draw snake
    for segment in snake.body:
        pygame.draw.rect(screen, GREEN, (segment[0]*GRID_SIZE, segment[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Draw food
    pygame.draw.rect(screen, RED, (food.position[0]*GRID_SIZE, food.position[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Draw obstacles
    for obstacle in obstacles.positions:
        pygame.draw.rect(screen, BLUE, (obstacle[0]*GRID_SIZE, obstacle[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Draw score and info
    font = pygame.font.Font(None, 24)
    score_text = font.render(f"Score: {snake.score}", True, WHITE)
    gen_text = font.render(f"Generation: {generation}", True, WHITE)
    genome_text = font.render(f"Genome: {genome_id}", True, WHITE)
    steps_text = font.render(f"Steps: {snake.steps}", True, WHITE)
    screen.blit(score_text, (10, 10))
    screen.blit(gen_text, (10, 40))
    screen.blit(genome_text, (10, 70))
    screen.blit(steps_text, (10, 100))

    pygame.display.flip()
    print(f"Drew game state: Gen {generation}, Genome {genome_id}, Score {snake.score}, Steps {snake.steps}")

def get_inputs(snake, food, obstacles):
    head = snake.body[0]

    # Direction to food
    food_dir_x = food.position[0] - head[0]
    food_dir_y = food.position[1] - head[1]

    # Normalize direction
    length = math.sqrt(food_dir_x**2 + food_dir_y**2)
    if length != 0:
        food_dir_x /= length
        food_dir_y /= length

    # Check for obstacles and body parts in 8 directions
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    vision = []
    for dx, dy in directions:
        found_object = False
        for distance in range(1, max(GRID_WIDTH, GRID_HEIGHT)):
            x = (head[0] + dx * distance) % GRID_WIDTH
            y = (head[1] + dy * distance) % GRID_HEIGHT
            if (x, y) in snake.body[1:]:
                vision.extend([1, 0, 1/distance])
                found_object = True
                break
            elif (x, y) in obstacles.positions:
                vision.extend([0, 1, 1/distance])
                found_object = True
                break
        if not found_object:
            vision.extend([0, 0, 0])

    # Current direction
    direction_one_hot = [0, 0, 0, 0]
    direction_index = [(0, -1), (0, 1), (-1, 0), (1, 0)].index(snake.direction)
    direction_one_hot[direction_index] = 1

    return [food_dir_x, food_dir_y] + vision + direction_one_hot

def eval_genomes(genomes, config):
    global generation
    generation += 1

    for genome_id, genome in genomes:
        print(f"Evaluating genome {genome_id} in generation {generation}")
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        snake = Snake()
        food = Food()
        obstacles = Obstacle()

        clock = pygame.time.Clock()

        for step in range(500):  # Increased step limit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # Get the inputs for the neural network
            inputs = get_inputs(snake, food, obstacles)

            # Get the output from the neural network
            output = net.activate(inputs)
            direction = output.index(max(output))

            # Update snake direction
            if direction == 0 and snake.direction != (0, 1):
                snake.direction = (0, -1)  # Up
            elif direction == 1 and snake.direction != (0, -1):
                snake.direction = (0, 1)   # Down
            elif direction == 2 and snake.direction != (1, 0):
                snake.direction = (-1, 0)  # Left
            elif direction == 3 and snake.direction != (-1, 0):
                snake.direction = (1, 0)   # Right

            # Move the snake
            snake.move()

            # Check if the snake ate the food
            if snake.body[0] == food.position:
                snake.grow()
                food.position = food.generate_position()
            else:
                snake.body.pop()

            # Check for collision with obstacles or self
            if (snake.check_collision() or
                    snake.body[0] in obstacles.positions or
                    snake.steps_without_food > 300):  # Increased tolerance
                print(f"Game over: Gen {generation}, Genome {genome_id}, Score {snake.score}, Steps {snake.steps}")
                break

            # Draw the game state
            draw_game(screen, snake, food, obstacles, generation, genome_id)
            clock.tick(60)  # Increased to 60 FPS

            if step % 50 == 0:
                print(f"Step {step}: Gen {generation}, Genome {genome_id}, Score {snake.score}")

        # Calculate fitness
        genome.fitness = snake.score * 10 + snake.steps / 25
        print(f"Fitness for genome {genome_id}: {genome.fitness}")

def run_neat(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes, 200)  # Increased to 200 generations

    print('\nBest genome:\n{!s}'.format(winner))

def main():
    global generation
    generation = 0

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run_neat(config_path)

if __name__ == "__main__":
    main()
#%%
