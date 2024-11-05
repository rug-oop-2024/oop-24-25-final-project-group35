from typing import Literal
from copy import deepcopy


class Feature:

    def __init__(self, name: str, type: Literal['numerical', 'categorical']):
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if isinstance(new_name, str) and new_name:
            self._name = new_name
        else:
            raise ValueError("Name must be a non-empty string.")

    @property
    def type(self) -> str:
        return deepcopy(self._type)

    def __str__(self):
        return f"Feature(name={self._name}, type={self._type})"

    def __repr__(self):
        return self.__str__()
