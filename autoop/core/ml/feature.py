from typing import Literal
from copy import deepcopy


class Feature:
    """Represents a feature in a dataset, with a name and a type
    (numerical or categorical).

    The Feature class provides controlled access to feature attributes,
    ensuring valid input for the feature name and secure retrieval of
    the feature type.
    """

    def __init__(
            self, name: str, type: Literal['numerical', 'categorical']
                ) -> None:
        """Initializes a Feature instance with a name and type.

        Args:
            name (str): The name of the feature.
            type (Literal['numerical', 'categorical']): The type of the feature
                either 'numerical' or 'categorical'.

        Raises:
            ValueError: If the provided name is not a valid non-empty string.
        """
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        """Gets the name of the feature.

        Returns:
            str: The name of the feature.
        """
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Sets a new name for the feature, ensuring it is a non-empty string.

        Args:
            new_name (str): The new name for the feature.

        Raises:
            ValueError: If the new name is not a valid non-empty string.
        """
        if isinstance(new_name, str) and new_name:
            self._name = new_name
        else:
            raise ValueError("Name must be a non-empty string.")

    @property
    def type(self) -> str:
        """Gets a copy of the feature type.

        Returns:
            str: The type of the feature ('numerical' or 'categorical').
        """
        return deepcopy(self._type)

    def __str__(self) -> str:
        """Provides a string representation of the Feature instance.

        Returns:
            str: A string in the format 'Feature(name=<name>, type=<type>)'.
        """
        return f"Feature(name={self._name}, type={self._type})"

    def __repr__(self) -> str:
        """Provides a detailed string representation of the Feature instance.

        Returns:
            str: A string in the format 'Feature(name=<name>, type=<type>)'.
        """
        return self.__str__()
