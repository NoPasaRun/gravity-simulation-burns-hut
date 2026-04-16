import math
import random

import numpy as np
import pygame
from pygame.locals import *
import pyrr

from core.camera import camera
from core.event import user_event
from core.gravity import GravityEngine
from core.handlers import handler
from core.model import OBJModel
from core.render import Renderer
from settings import screen_settings, camera_settings

from OpenGL.GL import *


POSITIONS, MASSES, VELOCITIES = [], [], []


def init(package: str, **kwargs) -> Renderer:
    positions, masses, velocities = [
        kwargs.pop(attr, [])
        for attr in ['positions', 'masses', 'velocities']
    ]
    count = len(positions) or len(masses) or len(velocities) or 1
    positions = positions or [[0, 0, 0]]*count

    POSITIONS.extend(positions)
    MASSES.extend(masses or [0]*count)
    VELOCITIES.extend(velocities or [[0, 0, 0]]*count)

    return Renderer(OBJModel(package), positions=np.array(positions), **kwargs)


def generate_asteroid_belt(n_asteroids=1000):
    asteroids_pos = []
    asteroids_vel = []
    asteroids_mass = []

    # Параметры пояса (в км)
    inner_radius = 329_000_000  # ~2.2 AU
    outer_radius = 478_000_000  # ~3.2 AU
    G = 6.674e-20
    M_sun = 1.989e30

    for _ in range(n_asteroids):
        # 1. Случайный радиус и угол
        r = random.uniform(inner_radius, outer_radius)
        angle = random.uniform(0, 2 * math.pi)

        # Небольшое отклонение по вертикали (толщина пояса)
        y = random.uniform(-5_000_000, 5_000_000)

        # Позиция (X, Y, Z)
        x = r * math.cos(angle)
        z = r * math.sin(angle)
        asteroids_pos.append([x, y, z])

        # 2. Орбитальная скорость v = sqrt(G * M / r)
        v_mag = math.sqrt((G * M_sun) / r)

        # Вектор скорости должен быть перпендикулярен вектору позиции (в плоскости XZ)
        vx = -v_mag * math.sin(angle)
        vz = v_mag * math.cos(angle)
        asteroids_vel.append([vx, 0, vz])

        # 3. Масса (небольшая, чтобы не влияли на планеты)
        asteroids_mass.append(random.uniform(1e10, 1e15))

    return init(
        "asteroids",
        scale=0.5,  # Совсем маленькие точки
        positions=asteroids_pos,
        velocities=asteroids_vel,
        masses=asteroids_mass,
        material_config={"Asteroid": {'type': 0, 'day': 'asteroid_tex.jpg'}}
    )


def main():
    pygame.init()
    pygame.font.init()
    # pygame.mixer.init()
    # pygame.mixer.music.load("src/song.mp3")
    pygame.display.set_mode(screen_settings.SCREEN, DOUBLEBUF | OPENGL)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    clock = pygame.time.Clock()

    proj_mat = pyrr.matrix44.create_perspective_projection_matrix(
        camera_settings.fov, screen_settings.ratio, camera_settings.near, camera_settings.far
    )

    g_models = [
        init(
            "sun", scale=1_391_400 / 20 * 50, masses=[1.989e30],
            material_config={
                "1": {'type': 0, 'day': 'texture.jpg'},
                "2": {'type': 0, 'day': 'texture.jpg'}
            }
        ),
        init(
            "mercury", scale=4879 / 1 * 1000, masses=[3.301e23],
            velocities=[[47.36, 0, 0]], positions=[[0, 0, 57_910_000]],
            material_config={"Material.001": {'type': 0, 'day': 'Mercury02.png'}}
        ),
        init(
            "venus", scale=12104 / 26.740545 * 1000, masses=[4.867e24],
            velocities=[[35.02, 0, 0]], positions=[[0, 0, 108_200_000]],
            material_config={
                "Material.001": {'type': 0, 'day': '8k_venus_surface.jpg'},
                'Material.002': {'type': 1, 'day': '4k_venus_atmosphere.jpg'}
            }
        ),
        init(
            "earth", scale=12742 / 2 * 1000, masses=[5.972e24],
            velocities=[[29.78, 0, 0]],
            positions=[[0, 0, 149_597_871]],
            material_config={'Material.002': {'type': 0, 'day': 'texture.jpg'}}
        ),
        init(
            "mars", scale=6779 / 5.092959 * 1000, masses=[6.39e23],
            velocities=[[24.07, 0, 0]], positions=[[0, 0, 227_900_000]],
            material_config={'mars': {'type': 0, 'day': 'mars.jpg'}}
        ),
        init(
            "jupiter", scale=139820 / 259.6871 * 1000, masses=[1.898e27],
            velocities=[[13.07, 0, 0]], positions=[[0, 0, 778_500_000]],
            material_config={"Material.001": {'type': 0, 'day': 'Jupiter_Map.jpeg'}}
        ),
        init(
            "saturn", scale=116460 / 3564.899 * 1000, masses=[5.683e26],
            velocities=[[9.68, 0, 0]], positions=[[0, 0, 1_433_000_000]],
            material_config={
                "saturn1_A": {'type': 0, 'day': 'Uv1_saturn1_diff.png'},
                'saturn2_A': {'type': 0, 'day': 'Uv1_saturn2_diff.png'},
                'saturn2_B': {'type': 1, 'day': 'Uv1_saturn2_diff.png'},
            }
        ),
        init(
            "uranus", scale=50724 / 3423.9346 * 1000, masses=[8.681e25],
            velocities=[[6.80, 0, 0]], positions=[[0, 0, 2_877_000_000]],
            material_config={
                'miranda': {'type': 0, 'day': 'Uv1_miranda1_diff.png'},
                "uranus1_A": {'type': 0, 'day': 'Uv1_uranus1_diff.png'},
                'uranus2_A': {'type': 0, 'day': 'Uv1_uranus2_diff.png'},
                'uranus2_B': {'type': 1, 'day': 'Uv1_uranus2_diff.png'},
            }
        ),
        init(
            "neptune", scale=49244 / 1181.3517 * 1000, masses=[1.024e26],
            velocities=[[5.43, 0, 0]], positions=[[0, 0, 4_503_000_000]],
            material_config={
                "neptune1_A": {'type': 0, 'day': 'Uv1_neptune1_diff.png'},
                'neptune2_A': {'type': 0, 'day': 'Uv1_neptune2_diff.png'},
                'neptune2_B': {'type': 1, 'day': 'Uv1_neptune2_diff.png'},
            }
        ),
        init(
            "pluto", scale=2376.6 / 2 * 1000, masses=[1.3e22],
            velocities=[[4.6691, 0, 0]], positions=[[0, 0,  5_900_000_000]],
            material_config={
                "Material.002": {"type": 0, "day": "pluto_texture_map_fixed_blur_by_bob3studio.jpg"},
            }
        )
        # generate_asteroid_belt(100_000)
    ]
    engine = GravityEngine(n=len(g_models))
    engine.set_initial_state(
        np.array(POSITIONS).astype(np.float32),
        np.array(VELOCITIES).astype(np.float32),
        np.array(MASSES).astype(np.float32),
    )

    while True:
        dt = clock.tick(screen_settings.FPS) / 1000

        for event in [*pygame.event.get(), user_event]:
            handler.call(event)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        view_matrix = camera.get_view_matrix()

        engine.update_physics(g_models, dt)
        for i, model in enumerate(g_models):
            model.render(view_matrix, proj_mat)

        pygame.display.set_caption(f"Universe Sim | FPS: {int(clock.get_fps())}")
        pygame.display.flip()


if __name__ == "__main__":
    main()
