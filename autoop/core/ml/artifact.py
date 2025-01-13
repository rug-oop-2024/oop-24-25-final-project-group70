from pydantic import BaseModel, Field
from typing import Dict, List, Any
import base64


class Artifact(BaseModel):
    """
    Represents a machine learning artifact, which could be a dataset, model,
    or other pipeline output. Each artifact is uniquely identified by an
    asset path and version.

    Attributes:
        asset_path (str): The storage path for the artifact.
        version (str): The artifact's version.
        data (bytes): Binary data representing the artifact's content.
        metadata (Dict[str, Any]): Metadata associated with the artifact,
            such as experiment ID and run ID.
        type (str): Type descriptor for the artifact (e.g., "model:torch").
        tags (List[str]): List of tags categorizing the artifact.
    """

    asset_path: str = Field(..., description="The path where the asset "
                            "is stored.")
    version: str = Field(..., description="The version of the artifact.")
    data: bytes = Field(..., description="The binary data representing "
                        "the artifact.")
    metadata: Dict[str, Any] = Field(default_factory=dict,
                                     description="Additional metadata for the "
                                     "artifact.")
    type: str = Field(...,
                      description="The type of artifact "
                      "(e.g., 'model:torch').")
    tags: List[str] = Field(default_factory=list,
                            description="Tags for categorization.")

    def get_id(self) -> str:
        """
        Generates a unique identifier for the artifact using a base64-encoded
        version of the asset path combined with the version.

        Returns:
            str: The unique identifier for the
                artifact in the format
                '{base64(asset_path)}:{version}'.
        """
        encoded_path = base64.urlsafe_b64encode(
            self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the artifact instance to a dictionary representation,
        suitable for storage or logging.

        Returns:
            Dict[str, Any]: A dictionary containing all
                            attributes of the artifact.
        """
        return {
            "asset_path": self.asset_path,
            "version": self.version,
            "data": self.data,
            "metadata": self.metadata,
            "type": self.type,
            "tags": self.tags,
        }

    def read(self) -> bytes:
        """
        Read data from the artifact.

        Returns:
            bytes: The binary data representing the artifact's content.
        """
        return self.data

    class Config:
        """Pydantic configuration for allowing bytes in 'data'."""
        arbitrary_types_allowed = True
