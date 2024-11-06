from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Custom exception raised when a specified path is not found."""

    def __init__(self, path: str) -> None:
        """Initializes the NotFoundError with a given path.

        Args:
            path (str): The path that could not be found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract base class that defines the interface for storage operations.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """Save data to a given path.

        Args:
            data (bytes): Data to be saved.
            path (str): The file path where data should be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """Load data from a specified path.

        Args:
            path (str): The file path from which to load data.

        Returns:
            bytes: The loaded data in bytes format.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete data at a specified path.

        Args:
            path (str): The file path of the data to delete.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """List all file paths under a specified directory.

        Args:
            path (str): The directory path to list files from.

        Returns:
            List[str]: A list of file paths.
        """
        pass


class LocalStorage(Storage):
    """Concrete implementation of Storage that handles local file operations.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """Initializes LocalStorage with a base directory path.

        Args:
            base_path (str): The base directory for storage.
            Defaults to "./assets".
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """Save data to a specified key within the base path.

        Args:
            data (bytes): The data to save.
            key (str): The unique identifier or file path for the data.
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """Load data from a specified key within the base path.

        Args:
            key (str): The unique identifier or file path for the data.

        Returns:
            bytes: The loaded data in bytes format.

        Raises:
            NotFoundError: If the specified path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """Delete data at the specified key within the base path.

        Args:
            key (str): The unique identifier or file path for the data.
            Defaults to root directory ("/").

        Raises:
            NotFoundError: If the specified path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        List all file paths under a specified directory prefix within
        the base path.

        Args:
            prefix (str): The directory prefix to list files from.

        Returns:
            List[str]: A list of file paths under the given prefix.

        Raises:
            NotFoundError: If the specified path does not exist.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        """Check if a path exists, and raise NotFoundError if it does not.

        Args:
            path (str): The path to check for existence.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """Construct an absolute path by joining the base
        path with the given path.

        Args:
            path (str): The relative path to join with the base path.

        Returns:
            str: The full absolute path.
        """
        return os.path.join(self._base_path, path)
