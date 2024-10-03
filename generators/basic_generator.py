from abc import ABC, abstractmethod
import numpy as np


class BaseGenerator(ABC):
    """
    Abstract base class for time series generators.
    """

    @abstractmethod
    def generate(self, length: int) -> np.ndarray:
        """
        Generate a time series of specified length.

        Args:
            length (int): Length of the time series to generate.

        Returns:
            np.ndarray: Generated time series data.
        """
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        """
        Set parameters for the generator.

        Args:
            **kwargs: Arbitrary keyword arguments for configuring the generator.
        """
        pass

    def __call__(self, length: int) -> np.ndarray:
        """
        Callable interface to generate a time series.
        """
        return self.generate(length)
