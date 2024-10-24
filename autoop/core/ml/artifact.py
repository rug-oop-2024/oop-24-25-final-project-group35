import base64

class Artifact:
    def __init__(self, name: str = "default.csv", asset_path: str = "some/path", version: str = "1.0.0", data: bytes = None, metadata: dict = None, tags: list = None, **kwargs):
        self.name = name  # Add a name attribute
        self.asset_path = asset_path
        self.version = version
        self.data = data or b""
        self.metadata = metadata or {}
        self.tags = tags or []  # Add a tags attribute
        self.type = kwargs.get('type', None)

    @property
    def id(self) -> str:
        encoded_path = base64.urlsafe_b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        return self.data

    def save(self, new_data: bytes) -> None:
        self.data = new_data

    def get_metadata(self) -> dict:
        return self.metadata

    def set_metadata(self, key: str, value) -> None:
        self.metadata[key] = value

    def __repr__(self):
        return f"Artifact(id={self.id}, name={self.name}, asset_path={self.asset_path}, version={self.version}, type={self.type}, tags={self.tags})"

'''
artifact = Artifact(
    asset_path="users/mo-assaf/models/yolov8.pth",
    version="1.0.2",
    data=b"binary_state_data",
    metadata={
        "experiment_id": "exp-123fbdiashdb",
        "run_id": "run-12378yufdh89afd"
    }
)

print(artifact.id)

# Access metadata
print(artifact.get_metadata())

# Save new data
artifact.save(b"new_binary_data")
print(artifact.read())
'''
