from __future__ import annotations

from dataclasses import dataclass
import pickle

import stan
from stan.fit import Fit
from stan.model import Model


@dataclass
class Stan:
    """Bundle a Stan fit with its model."""

    model: Model
    fit: Fit

    def save(self, file: str) -> None:
        """Save Stan model and fit using supplied file name."""
        with open(file + ".pkl", "wb") as f:
            pickle.dump(self, f, protocol=-1)

    @staticmethod
    def load(file: str) -> Stan:
        """Load Stan model and fit using supplied file name."""
        with open(file + ".pkl", "rb") as f:
            return pickle.load(f)

    def equal(self, other: Stan) -> bool:
        """For now just compare fit for (str) equality. Model name seems to vary unpredictably."""
        return str(self.fit) == str(other.fit)
