import sys
from datetime import datetime
from typing import Set

import pygame as pg
from core.camera import camera
from core.event import EventHandler, UserEvent
from core.gravity import GravityEngine
from settings import screen_settings

handler = EventHandler()
PRESSED = {'last': datetime.now(), 'key': None, 'pass': False}


@handler.set(pg.QUIT)
def exit_handler(*_):
    sys.exit(0)


@handler.set(pg.MOUSEMOTION)
def cursor_handler(event):
    camera.handle_mouse(*event.rel)
    pg.mouse.set_pos((screen_settings.SCREEN[0] // 2, screen_settings.SCREEN[1] // 2))


DIRECTIONS = {
    pg.K_w: camera.move_forward,
    pg.K_s: lambda: camera.move_forward(True),
    pg.K_d: camera.move_right,
    pg.K_a: lambda: camera.move_right(True),
    pg.K_SPACE: camera.move_up,
    pg.K_LSHIFT: lambda: camera.move_up(True),
}


ACTIVE_KEYS: Set[int] = set()


@handler.set(pg.KEYDOWN)
def move_handler(event):
    if event.key == pg.K_0:
        GravityEngine.ENABLED = not GravityEngine.ENABLED
    elif event.key == pg.K_ESCAPE:
        sys.exit(0)
    elif event.key == pg.K_MINUS:
        GravityEngine.TIME /= 10
    elif event.key == pg.K_EQUALS:
        GravityEngine.TIME *= 10

    if event.key in DIRECTIONS:
        ACTIVE_KEYS.add(event.key)
        if event.key == PRESSED['key']:
            PRESSED['pass'] = True
        PRESSED['last'], PRESSED['key'] = datetime.now(), event.key


@handler.set(pg.KEYUP)
def release_handler(event):
    if event.key in DIRECTIONS:
        ACTIVE_KEYS.discard(event.key)

    if (datetime.now() - PRESSED['last']).microseconds > 250_000:
        PRESSED['key'], PRESSED['pass'] = None, False


@handler.set(UserEvent.type)
def update_handler(*_):
    if PRESSED['pass']:
        camera.velocity = min(camera.MAX_VELOCITY, camera.velocity ** 2)

    for key in ACTIVE_KEYS:
        if key in DIRECTIONS:
            DIRECTIONS[key]()
