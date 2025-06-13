import numpy as np
import random
from PIL import Image
import os


class NavigationEnvironment:
    def __init__(self, image_files, centers_dict, view_size=(64, 64), max_steps=100, history_length=5):
        self.image_files = image_files
        self.centers_dict = centers_dict
        self.view_size = view_size
        self.max_steps = max_steps
        self.history_length = history_length
        
        self.num_actions = 5
        self.move_step = 20
        
        self.current_image = None
        self.current_image_path = None
        self.current_center = None
        self.current_position = None
        self.current_view = None
        self.steps_taken = 0
        
        self.position_history = []
        self.velocity = (0, 0)

    def reset(self):
        self.current_image_path = random.choice(self.image_files)
        self.current_image = Image.open(self.current_image_path).convert('L')
        
        filename = os.path.basename(self.current_image_path)
        self.current_center = self.centers_dict.get(filename, (128, 128))
        
        img_width, img_height = self.current_image.size
        margin = max(self.view_size) // 2
        
        x = random.randint(margin, img_width - margin)
        y = random.randint(margin, img_height - margin)
        self.current_position = (x, y)
        
        self.steps_taken = 0
        self.position_history = [self.current_position]
        self.velocity = (0, 0)
        
        self.current_view = self._get_current_view()
        return self.current_view

    def step(self, action):
        prev_distance = self._calculate_distance_to_center()
        
        self._apply_action(action)
        self.steps_taken += 1
        
        self.position_history.append(self.current_position)
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
        
        self.current_view = self._get_current_view()
        
        distance = self._calculate_distance_to_center()
        reward = self._calculate_reward(distance)
        
        done = distance < 10 or self.steps_taken >= self.max_steps
        
        info = {
            'distance': distance,
            'position': self.current_position,
            'center': self.current_center,
            'steps': self.steps_taken
        }
        
        return self.current_view, reward, done, info

    def _apply_action(self, action):
        x, y = self.current_position
        vx, vy = self.velocity
        
        momentum_factor = 0.3
        
        if action == 0:
            new_vx = -self.move_step + momentum_factor * vx
            new_vy = momentum_factor * vy
        elif action == 1:
            new_vx = momentum_factor * vx
            new_vy = -self.move_step + momentum_factor * vy
        elif action == 2:
            new_vx = self.move_step + momentum_factor * vx
            new_vy = momentum_factor * vy
        elif action == 3:
            new_vx = momentum_factor * vx
            new_vy = self.move_step + momentum_factor * vy
        else:
            new_vx = momentum_factor * vx
            new_vy = momentum_factor * vy
        
        new_x = x + int(new_vx)
        new_y = y + int(new_vy)
        
        img_width, img_height = self.current_image.size
        margin = max(self.view_size) // 2
        
        new_x = max(margin, min(new_x, img_width - margin))
        new_y = max(margin, min(new_y, img_height - margin))
        
        self.current_position = (new_x, new_y)
        self.velocity = (new_vx, new_vy)

    def _get_current_view(self):
        x, y = self.current_position
        view_w, view_h = self.view_size
        
        left = x - view_w // 2
        top = y - view_h // 2
        right = left + view_w
        bottom = top + view_h
        
        view = self.current_image.crop((left, top, right, bottom))
        view = view.resize(self.view_size)
        
        view_array = np.array(view, dtype=np.float32) / 255.0
        return view_array.reshape(1, view_h, view_w)

    def _calculate_distance_to_center(self):
        return np.sqrt(
            (self.current_position[0] - self.current_center[0]) ** 2 +
            (self.current_position[1] - self.current_center[1]) ** 2
        )

    def _calculate_reward(self, distance):
        distance_reward = -0.001 * (distance ** 2)
        
        if distance < 5:
            distance_reward += 20.0
        elif distance < 10:
            distance_reward += 10.0 * (1 - (distance - 5) / 5)
        elif distance < 20:
            distance_reward += 5.0 * (1 - (distance - 10) / 10)
        
        oscillation_penalty = 0
        if self._is_oscillating():
            oscillation_penalty = -5.0
        
        progress_reward = 0
        if len(self.position_history) > 1:
            prev_distance = np.sqrt(
                (self.position_history[-2][0] - self.current_center[0]) ** 2 +
                (self.position_history[-2][1] - self.current_center[1]) ** 2
            )
            progress_reward = (prev_distance - distance) * 0.5
        
        step_penalty = -0.05
        
        return distance_reward + step_penalty + oscillation_penalty + progress_reward

    def _is_oscillating(self):
        if len(self.position_history) < 4:
            return False
        
        recent_positions = self.position_history[-4:]
        
        for i in range(len(recent_positions) - 2):
            for j in range(i + 2, len(recent_positions)):
                pos1 = recent_positions[i]
                pos2 = recent_positions[j]
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if distance < 15:
                    return True
        
        return False
