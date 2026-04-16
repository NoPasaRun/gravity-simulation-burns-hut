from typing import List

import numpy as np
import pyrr

from settings import camera_settings


class Camera:

    MAX_VELOCITY = 100_000_000

    def __init__(self, pos=None, rotation_axis=None, speed=1.0):
        self.position = np.array(pos or [0, 0, 0], dtype=np.float32)
        self.__rotation_axis = np.array(rotation_axis or [0, 1, 0], dtype=np.float32)
        self.__rotation_axis /= np.linalg.norm(self.__rotation_axis)
        self.speed, self.velocity = speed, speed
        self.__forward = np.array([0, 0, -1], dtype=np.float32)
        self.__right = np.array([1, 0, 0], dtype=np.float32)
        self.__up = np.array([0, 1, 0], dtype=np.float32)

    @property
    def up(self):
        return self.__up

    @property
    def forward(self):
        return self.__forward

    @property
    def right(self):
        return self.__right

    def _update_base_axis(self):
        self.__right = pyrr.vector.normalise(np.cross(self.__forward, self.rotation_axis))
        self.__up = pyrr.vector.normalise(np.cross(self.__right, self.__forward))

    @property
    def rotation_axis(self):
        return self.__rotation_axis

    def set_rotation_axis(self, new_axis: List, smooth: bool = True):
        new_axis = np.array(new_axis, dtype=np.float32)
        new_axis = new_axis / np.linalg.norm(new_axis)

        if smooth:
            self.__rotation_axis = self.rotation_axis * (1 - 0.1) + new_axis * 0.1
        else:
            self.__rotation_axis = new_axis
        self.__rotation_axis /= np.linalg.norm(self.rotation_axis)
        self._update_base_axis()

    def _project_on_main(self, vec):
        """Проекция вектора на плоскость main (перпендикулярную rotation_axis)"""
        proj = vec - np.dot(vec, self.rotation_axis) * self.rotation_axis
        norm = np.linalg.norm(proj)
        return proj / norm if norm > 1e-6 else proj

    def handle_mouse(self, dx, dy):
        sens = camera_settings.sens

        # 1. Yaw: Вращение вокруг вектора наклона (rotation_axis)
        yaw_rot = pyrr.matrix33.create_from_axis_rotation(self.rotation_axis, -dx * sens)

        # 2. Pitch: Вращение вокруг локального Right
        # Сначала найдем текущий Right как перпендикуляр к взгляду и оси наклона
        current_right = np.cross(self.__forward, self.rotation_axis)
        r_norm = np.linalg.norm(current_right)

        # Если смотрим в полюс, используем сохраненный right
        current_right = self.__right if r_norm < 1e-6 else current_right / r_norm

        pitch_rot = pyrr.matrix33.create_from_axis_rotation(current_right, -dy * sens)

        # Применяем вращения
        new_forward = yaw_rot @ (pitch_rot @ self.__forward)
        new_forward /= np.linalg.norm(new_forward)

        # 3. Лимит Pitch (Clipping)
        # Угол между forward и rotation_axis должен быть в пределах (5, 175) градусов
        dot = np.dot(new_forward, self.rotation_axis)
        if abs(dot) < 0.995:  # Не даем смотреть строго в полюс (защита от исчезновения)
            self.__forward = new_forward
        self._update_base_axis()

    def handle_z_roll(self, roll):
        roll_angle = roll * camera_settings.sens * 20
        roll_rot = pyrr.matrix33.create_from_axis_rotation(self.__forward, roll_angle)
        self.set_rotation_axis(pyrr.vector.normalise(roll_rot @ self.rotation_axis), smooth=False)

    def handle_x_roll(self, roll):
        angle = roll * camera_settings.sens * 20
        rotation_matrix = pyrr.matrix33.create_from_axis_rotation(self.__right, angle)

        self.__rotation_axis = pyrr.matrix33.apply_to_vector(rotation_matrix, self.rotation_axis)
        self.__forward = pyrr.matrix33.apply_to_vector(rotation_matrix, self.__forward)

        self.__rotation_axis = pyrr.vector.normalise(self.rotation_axis)
        self.__forward = pyrr.vector.normalise(self.__forward)
        self.__up = pyrr.vector.normalise(np.cross(self.__right, self.__forward))

    def move_forward(self, rev=False):
        move_dir = self._project_on_main(self.__forward)
        self.position += move_dir * self.velocity * (-1 if rev else 1)

    def move_right(self, rev=False):
        self.position += self.__right * self.velocity * (-1 if rev else 1)

    def move_up(self, rev=False):
        self.position += self.rotation_axis * self.velocity * (-1 if rev else 1)

    def get_view_matrix(self):
        # 1. Создаем матрицу чистого вращения из векторов ориентации
        # Мы собираем её из Right, Up и -Forward
        rotation = np.eye(4, dtype=np.float32)
        rotation[0, :3] = self.__right
        rotation[1, :3] = self.__up
        rotation[2, :3] = -self.__forward

        # ТРАНСПОНИРУЕМ, чтобы векторы стали столбцами (стандарт OpenGL/Pyrr)
        # Это ключевой момент, который убирает "вращение вокруг модели"
        rotation = rotation.T

        # 2. Создаем матрицу трансляции на инвертированную позицию
        translation = pyrr.matrix44.create_from_translation(-self.position)

        # 3. Итоговая View Matrix = Rotation * Translation
        # ВАЖНО: порядок умножения определяет, вращаемся мы на месте или по орбите
        return pyrr.matrix44.multiply(translation, rotation)


camera = Camera(speed=10_000, pos=[0, 500_000_000, 0])
