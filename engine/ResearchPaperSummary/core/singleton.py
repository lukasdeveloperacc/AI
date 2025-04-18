from typing import TypeVar, Type

T = TypeVar("T", bound="Singleton")


class Singleton:
    _instances = {}

    @classmethod
    def instance(cls: Type[T], *args, **kwargs) -> T:
        if cls not in cls._instances:
            cls._instances[cls] = cls(*args, **kwargs)

        return cls._instances[cls]

    @classmethod
    def reset_instance(cls):
        if cls in cls._instances:
            del cls._instances[cls]
