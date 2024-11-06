from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """Represents a dataset artifact that can read and save data in CSV format.

    The Dataset class provides methods for saving a DataFrame as a CSV file
    and reading a saved CSV file back into a DataFrame.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the Dataset instance with a dataset type.

        Args:
            *args: Variable length argument list for additional parameters.
            **kwargs: Arbitrary keyword arguments for additional parameters.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0") -> "Dataset":
        """Creates a Dataset instance from a DataFrame.

        Args:
            data (pd.DataFrame): The data to be saved as a dataset.
            name (str): The name of the dataset.
            asset_path (str): The path where the dataset will be saved.
            version (str, optional): The version of the dataset.
            Defaults to "1.0.0".

        Returns:
            Dataset: A new Dataset instance with the provided data
            saved in CSV format.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """Reads the dataset from storage and returns it as a DataFrame.

        Returns:
            pd.DataFrame: The dataset loaded from storage.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Saves a DataFrame to storage in CSV format.

        Args:
            data (pd.DataFrame): The DataFrame to be saved.

        Returns:
            bytes: The encoded CSV data.

        Raises:
            Exception: If there is an error during the saving process.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
