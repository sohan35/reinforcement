import pygame
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tic_tac_toe_env import TicTacToeEnv

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

def draw_board(screen, board):
    # Draw the Tic Tac Toe board
    screen.fill(BLACK)
    # Horizontal lines
    for y in range(1, 3):
        pygame.draw.line(screen, WHITE, (100, y * 200), (500, y * 200), 5)
    # Vertical lines
    for x in range(1, 3):
        pygame.draw.line(screen, WHITE, (x * 200, 100), (x * 200, 500), 5)

    # Draw X's and O's
    for i in range(3):
        for j in range(3):
            if board[i][j] == 1:
                pygame.draw.line(screen, RED, (j * 200 + 50, i * 200 + 50), (j * 200 + 150, i * 200 + 150), 10)
                pygame.draw.line(screen, RED, (j * 200 + 150, i * 200 + 50), (j * 200 + 50, i * 200 + 150), 10)
            elif board[i][j] == -1:
                pygame.draw.circle(screen, BLUE, (j * 200 + 100, i * 200 + 100), 70, 10)

def display_text(screen, text, color, font_size, x, y):
    # Display text on the screen
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(x, y))
    screen.blit(text_surface, text_rect)

def evaluate_agent():
    # Load the trained agent
    model = PPO.load("tic_tac_toe_model")

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Tic Tac Toe")

    # Create Tic Tac Toe environment
    env = TicTacToeEnv()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN and not done:
                mouseX, mouseY = pygame.mouse.get_pos()
                row = mouseY // 200
                col = mouseX // 200
                action = row * 3 + col
                if env.board[row][col] == 0:  # Check if the move is valid
                    obs, reward, done, _ = env.step(action)
                    if not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, _ = env.step(action)
                        draw_board(screen, env.board)  # Draw the board after the agent's move
                        pygame.display.update()  # Update the display
                        time.sleep(1)  # Add a short delay
                if done:
                    break

        draw_board(screen, env.board)
        pygame.display.update()

        # Check if the game ended in a draw
        if env._is_board_full() and not done:
            done = True
            display_text(screen, "It's a draw!", WHITE, 36, 300, 300)
            pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    evaluate_agent()
