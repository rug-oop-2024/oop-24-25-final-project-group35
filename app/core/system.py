from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """Handles the registration, retrieval, and management of artifacts.

    It provides methods to register, list all artifacts,
    retrieve a specific artifact by its ID, and delete artifacts.
    """

    def __init__(self, database: Database, storage: Storage) -> None:
        """Initializes the ArtifactRegistry with a database and storage system.

        Args:
            database (Database): The database to store artifact metadata.
            storage (Storage): The storage system for saving artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """Registers an artifact by saving its data and metadata.

        Args:
            artifact (Artifact): The artifact to register.
        """
        # Save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # Save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """Lists all registered artifacts, optionally filtering by type.

        Args:
            type (str, optional): The type of artifacts to filter by.

        Returns:
            List[Artifact]: A list of Artifact objects.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """Retrieves an artifact by its ID.

        Args:
            artifact_id (str): The ID of the artifact to retrieve.

        Returns:
            Artifact: The retrieved artifact.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """Deletes an artifact by its ID from both the database and storage.

        Args:
            artifact_id (str): The ID of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """Singleton class that manages the AutoML system.

    The AutoMLSystem provides a unified interface to access and manage the
    artifact registry, storage, and database.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """Initializes the AutoMLSystem with a local storage and database.

        Args:
            storage (LocalStorage): The storage system for artifact data.
            database (Database): The database system for artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """Retrieves the singleton instance of the AutoMLSystem.

        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"), 
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """Provides access to the artifact registry.

        Returns:
            ArtifactRegistry: The artifact registry associated with the system.
        """
        return self._registry
