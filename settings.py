from dataclasses import dataclass


@dataclass
class CameraSettings:
    fov: float = 60
    near: float = 0.1
    far: float = 10**10
    sens: float = 0.001


@dataclass
class ScreenSettings:
    FPS: int = 60
    SCREEN: tuple = (1920, 1080)
    ratio: float = SCREEN[0] / SCREEN[1]


camera_settings = CameraSettings()
screen_settings = ScreenSettings()
