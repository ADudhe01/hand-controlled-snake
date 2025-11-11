import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os

# =======================
# --- Mediapipe Setup ---
# =======================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# =======================
# --- Snake Game Logic ---
# =======================
class SnakeGame:
    def __init__(self, w=800, h=600, cell_size=20):
        self.w = w
        self.h = h
        self.cell = cell_size
        self.high_score_file = "highscore.txt"
        self.high_score = self.load_high_score()
        self.reset()

    def load_high_score(self):
        """Load saved high score from file"""
        if os.path.exists(self.high_score_file):
            try:
                with open(self.high_score_file, "r") as f:
                    return int(f.read().strip())
            except:
                return 0
        return 0

    def save_high_score(self):
        """Save current high score to file"""
        with open(self.high_score_file, "w") as f:
            f.write(str(self.high_score))

    def reset(self):
        self.snake = [(self.w // 2, self.h // 2)]
        self.direction = (self.cell, 0)
        self.spawn_food()
        self.game_over = False
        self.score = 0

    def spawn_food(self):
        self.food = (random.randrange(0, self.w, self.cell),
                     random.randrange(0, self.h, self.cell))

    def change_direction(self, new_dir):
        opposite = (-self.direction[0], -self.direction[1])
        if new_dir != opposite:
            self.direction = new_dir

    def update(self):
        if self.game_over:
            return

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = ((head_x + dx) % self.w, (head_y + dy) % self.h)

        # ✅ Prevent early self-collision (only check if length > 4)
        if len(self.snake) > 4 and new_head in self.snake:
            if self.score > self.high_score:
                self.high_score = self.score
                self.save_high_score()
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        # Eat food
        if abs(new_head[0] - self.food[0]) < self.cell and abs(new_head[1] - self.food[1]) < self.cell:
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()

    def draw(self, img):
        # Draw food
        cv2.rectangle(img, self.food,
                      (self.food[0] + self.cell, self.food[1] + self.cell),
                      (0, 0, 255), cv2.FILLED)

        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i > 0 else (0, 150, 255)
            cv2.rectangle(img, (x, y), (x + self.cell, y + self.cell), color, cv2.FILLED)

        # Draw current score (top left)
        cv2.putText(img, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw high score (top right)
        output = self.high_score if self.high_score > self.score else self.score
        text = f"High Score: {output}"
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(img, text, (self.w - text_w - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Game over screen
        if self.game_over:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text1 = "GAME OVER"
            scale1, thick1 = 2, 5
            (w1, h1), _ = cv2.getTextSize(text1, font, scale1, thick1)
            cv2.putText(img, text1,
                        ((self.w - w1)//2, (self.h//2)),
                        font, scale1, (0, 0, 255), thick1)

            text2 = "Press R to Restart"
            scale2, thick2 = 1, 2
            (w2, h2), _ = cv2.getTextSize(text2, font, scale2, thick2)
            cv2.putText(img, text2,
                        ((self.w - w2)//2, (self.h//2) + h1 + 30),
                        font, scale2, (255, 255, 255), thick2)


# ==========================
# --- Main Function ---
# ==========================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS camera fix
    game = SnakeGame()
    speed = 0.1
    last_move = time.time()

    window_name = "Hand Controlled Snake (Split Screen)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 600)

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Camera not detected")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        direction = None
        cx, cy = game.w // 2, game.h // 2

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark[8]  # Index fingertip
                x, y = int(lm.x * game.w), int(lm.y * game.h)

                # Draw fingertip + center reference
                cv2.circle(frame, (cx, cy), 10, (255, 255, 0), 2)
                cv2.circle(frame, (x, y), 15, (0, 255, 255), cv2.FILLED)

                dx, dy = x - cx, y - cy
                if abs(dx) > abs(dy):
                    if dx > 50:
                        direction = (game.cell, 0)
                    elif dx < -50:
                        direction = (-game.cell, 0)
                else:
                    if dy > 50:
                        direction = (0, game.cell)
                    elif dy < -50:
                        direction = (0, -game.cell)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Apply direction changes
        if direction:
            game.change_direction(direction)

        # Update snake movement
        if time.time() - last_move > speed:
            game.update()
            last_move = time.time()

        # Create split-screen display
        board = np.zeros((game.h, game.w, 3), dtype=np.uint8)
        game.draw(board)
        frame_resized = cv2.resize(frame, (game.w, game.h))
        combined = np.hstack((frame_resized, board))

        cv2.imshow(window_name, combined)

        # Controls
        key = cv2.waitKey(10)
        if key != -1:
            k = chr(key & 0xFF).lower()
            if k == 'r':
                game.reset()
            elif k == 'q':
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()