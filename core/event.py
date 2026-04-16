from typing import Dict, List, Callable


class UserEvent:
    type: int = -1010

class EventHandler:
    __handlers: Dict[int, List[Callable]] = dict()

    def __new__(cls, *args):
        return cls

    @classmethod
    def call(cls, event, *args, **kwargs):
        for callback in cls.__handlers.get(event.type, []):
            callback(event, *args, **kwargs)

    @classmethod
    def set(cls, event_type: int):
        def decorator(func: Callable):
            if not (callbacks := cls.__handlers.get(event_type, [])):
                cls.__handlers[event_type] = callbacks
            callbacks.append(func)
            return func
        return decorator


user_event = UserEvent()
