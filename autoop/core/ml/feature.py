from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd


class Feature(BaseModel):
    """
    Represents a feature within a dataset, including its name, type,
    and relevant statistics.

    Attributes:
        name (str): The name of the feature.
        type (Literal['categorical', 'numerical']): The type of the feature.
        mean (Optional[float]): The mean of the feature, applicable
                                for numerical types.
        unique_values (Optional[int]): The count of unique values,
                                       applicable for categorical types.
    """

    name: str = Field(..., description="The name of the feature.")
    type: Literal['categorical', 'numerical'] = Field(
        ..., description="The type of the feature.")
    mean: Optional[float] = Field(None,
                                  description="Mean of the feature, "
                                  "for numerical types only.")
    unique_values: Optional[int] = Field(None,
                                         description="Number of unique "
                                         "values, for categorical types only.")

    def compute_statistics(self, data: pd.Series) -> None:
        """
        Computes and stores relevant statistics for
        the feature based on its type.

        Args:
            data (pd.Series): A pandas Series containing
            the data for this feature.
        """
        if self.type == 'numerical':
            self.mean = data.mean()
        elif self.type == 'categorical':
            self.unique_values = data.nunique()

    def __str__(self) -> str:
        """
        Returns a string representation of the feature, including name, type,
        and any computed statistics.

        Returns:
            str: A string describing the feature.
        """
        if self.type == 'numerical' and self.mean is not None:
            return (f"Feature(name={self.name}, type={self.type}, "
                    f"mean={self.mean:.2f})")
        elif self.type == 'categorical' and self.unique_values is not None:
            return (f"Feature(name={self.name}, type={self.type}, "
                    f"unique_values={self.unique_values})")
        else:
            return f"Feature(name={self.name}, type={self.type})"

    class Config:
        arbitrary_types_allowed = True
