from typing import Optional
import numpy as np


class WeightsCalculator:
    def __init__(self):
        pass

    def calculate_weights(self, n: int, n_bands: int, grid_band_index: int) -> np.ndarray:
        """ Calculate inverse weights based on the grid band index

        Args:
            n (int): Number of nodes/elements
            n_bands (int): Total number of grid bands
            grid_band_index (int): Index of the grid band

        Returns:
            Array of weights.
        """
        raise NotImplementedError("Subclasses must implement calculate_weights method")


class LinearWeightsCalculator(WeightsCalculator):
    """ Linear weights calculator for different grid bands within a nest

    """
    def __init__(self):
        super().__init__()

    def calculate_weights(self, n: int, n_bands: int, grid_band_index: int) -> np.ndarray:
        """ Calculate linear weights based on the grid band index

        Args:
            n (int): Number of nodes/elements
            n_bands (int): Total number of grid bands
            grid_band_index (int): Index of the grid band

        Returns:
            Array of weights.
        """
        if n_bands <= 1:
            return np.ones(n)
        else:
            # Linear decay: weight = 1 - (grid_band_index / n_bands)
            # This ensures weight = 1.0 for index 0, and weight = 0.0 for index n_bands
            weight = 1.0 - (grid_band_index / n_bands)
            return np.ones(n) * weight


class InverseWeightsCalculator(WeightsCalculator):
    """ Inverse weights calculator for different grid bands within a nest

    """
    def __init__(self):
        super().__init__()

    def calculate_weights(self, n: int, n_bands: int, grid_band_index: int) -> np.ndarray:
        """ Calculate inverse weights based on the grid band index

        Args:
            n (int): Number of nodes/elements
            n_bands (int): Total number of grid bands
            grid_band_index (int): Index of the grid band

        Returns:
            Array of weights.
        """
        weight = 1.0 / (grid_band_index+1.0)
        return np.ones(n) * weight


class ExponentialWeightsCalculator(WeightsCalculator):
    """ Exponential weights calculator for different grid bands within a nest

    """
    def __init__(self):
        super().__init__()

    def calculate_weights(self, n: int, n_bands: int, grid_band_index: int) -> np.ndarray:
        """ Calculate exponential weights based on the grid band index

        Args:
            n (int): Number of nodes/elements
            n_bands (int): Total number of grid bands
            grid_band_index (int): Index of the grid band

        Returns:
            Array of weights.
        """
        weight = np.exp(-grid_band_index)
        return np.ones(n) * weight


def get_weights_calculator(calculation_method: str) -> WeightsCalculator:
    """ Factory function to get the appropriate WeightsCalculator instance.

    Args:
        calculation_method (str): The method of weight calculation ("linear", "inverse", "exponential").

    Returns:
        WeightsCalculator: An instance of the corresponding WeightsCalculator subclass.

    Raises:
        ValueError: If an unknown calculation method is provided.
    """
    if calculation_method == "linear":
        return LinearWeightsCalculator()
    elif calculation_method == "inverse":
        return InverseWeightsCalculator()
    elif calculation_method == "exponential":
        return ExponentialWeightsCalculator()
    else:
        raise ValueError(f"Unknown weights calculation method: {calculation_method}")
