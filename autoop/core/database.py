import json
from typing import Tuple, List, Union
from autoop.core.storage import Storage


class Database:
    """A database class that stores and manages data in collections
    and persists data to external storage.

    The Database class allows for setting, retrieving, deleting,
    and listing entries within collections, with data persisted
    to a specified storage backend.
    """

    def __init__(self, storage: Storage) -> None:
        """Initializes the Database with a specified storage backend.

        Args:
            storage (Storage): The storage backend to use for persistence.
        """
        self._storage = storage
        self._data = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """Store or update an entry in a specified collection.

        Args:
            collection (str): The name of the collection to store data in.
            id (str): The unique identifier for the data entry.
            entry (dict): The data entry to store.

        Returns:
            dict: The data entry that was stored.

        Raises:
            AssertionError: If entry is not a dictionary, collection
                            is not a string, or id is not a string.
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"
        if not self._data.get(collection, None):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """Retrieve an entry from a specified collection.

        Args:
            collection (str): The name of the collection to retrieve data from.
            id (str): The unique identifier for the data entry.

        Returns:
            Union[dict, None]: The data entry if found, or None if
            it doesn't exist.
        """
        if not self._data.get(collection, None):
            return None
        return self._data[collection].get(id, None)

    def delete(self, collection: str, id: str) -> None:
        """Delete an entry from a specified collection.

        Args:
            collection (str): The name of the collection to delete data from.
            id (str): The unique identifier for the data entry.

        Returns:
            None
        """
        if not self._data.get(collection, None):
            return
        if self._data[collection].get(id, None):
            del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """List all entries in a specified collection.

        Args:
            collection (str): The name of the collection to list entries from.

        Returns:
            List[Tuple[str, dict]]: A list of tuples containing the id
            and data for each entry in the collection, or an empty list if the
            collection doesn't exist.
        """
        if not self._data.get(collection, None):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self) -> None:
        """Reload the database from the storage backend.

        This clears and reinitializes the database state based
        on the current data in storage.
        """
        self._load()

    def _persist(self) -> None:
        """Persist the current database state to the storage backend.

        This method saves each entry in the database to storage and removes
        entries from storage that are no longer in the idatabase.
        """
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                self._storage.save(
                    json.dumps(item).encode(), f"{collection}/{id}"
                )

        # Remove entries from storage that were deleted from the database
        keys = self._storage.list("")
        for key in keys:
            collection, id = key.split("/")[-2:]
            if not self._data.get(collection, id):
                self._storage.delete(f"{collection}/{id}")

    def _load(self) -> None:
        """Load the database state from the storage backend.

        This method initializes the database with entries
        stored in the storage.
        """
        self._data = {}
        for key in self._storage.list(""):
            collection, id = key.split("/")[-2:]
            data = self._storage.load(f"{collection}/{id}")
            # Ensure the collection exists in the dictionary
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][id] = json.loads(data.decode())
