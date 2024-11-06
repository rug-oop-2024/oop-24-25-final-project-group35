import base64


class Artifact:
    """Represents a data artifact with metadata, tags, and a versioned ID.

    The Artifact class provides methods for reading, saving, and managing
    metadata associated with data artifacts, which are identified by a unique,
    encoded ID.
    """

    def __init__(self, name: str = "default.csv",
                 asset_path: str = "some/path", version: str = "1.0.0",
                 data: bytes = None, metadata: dict = None,
                 tags: list = None, **kwargs) -> None:
        """Initializes an Artifact instance with optional metadata, tags
        and data.

        Args:
            name (str): The name of the artifact. Defaults to "default.csv".
            asset_path (str): The path to the artifact asset.
            Defaults to "some/path".
            version (str): The version of the artifact. Defaults to "1.0.0".
            data (bytes): The binary data for the artifact. Defaults to None.
            metadata (dict): Additional metadata for the artifact.
            Defaults to None.
            tags (list): Tags for categorizing the artifact. Defaults to None.
            **kwargs: Additional keyword arguments,
            including 'type' for the artifact type.
        """
        self.name = name
        self.asset_path = asset_path
        self.version = version
        self.data = data or b""
        self.metadata = metadata or {}
        self.tags = tags or []
        self.type = kwargs.get('type', None)

    @property
    def id(self) -> str:
        """Generates a unique, encoded ID based on the asset path and version.

        Returns:
            str: The unique ID for the artifact, encoded with the asset
            path and version.
        """
        encoded_path = base64.urlsafe_b64encode(
            self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """Reads the artifact's data.

        Returns:
            bytes: The binary data of the artifact.
        """
        return self.data

    def save(self, new_data: bytes) -> None:
        """Saves new data to the artifact.

        Args:
            new_data (bytes): The binary data to be saved in the artifact.
        """
        self.data = new_data

    def get_metadata(self) -> dict:
        """Retrieves the metadata associated with the artifact.

        Returns:
            dict: The metadata dictionary for the artifact.
        """
        return self.metadata

    def set_metadata(self, key: str, value: str) -> None:
        """Sets a metadata key-value pair for the artifact.

        Args:
            key (str): The metadata key.
            value: The value associated with the key.
        """
        self.metadata[key] = value

    def __repr__(self) -> str:
        """Provides a string representation of the Artifact instance.

        Returns:
            str: A formatted string displaying the artifact's details.
        """
        return (f"Artifact(id={self.id}, name={self.name}, "
                f"asset_path={self.asset_path}, version={self.version}, "
                f"type={self.type}, tags={self.tags})")
