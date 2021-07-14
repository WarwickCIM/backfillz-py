from dataclasses import dataclass
from math import floor
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from stan.fit import Fit  # type: ignore


@dataclass
class MCMCRun:
    """A Stan fit, plus some derived data."""

    samples: Fit

    def iter_chains(self, param: str, index: int = 0) -> np.ndarray:
        """Return (n_chains × n_samples) matrix of draws for a given parameter."""
        n_chains, n_samples = self.samples.num_chains, self.samples.num_samples
        xss = np.zeros((n_chains, n_samples))
        for n in range(0, n_chains):
            xss[n] = self.samples[param][index][n * n_samples: (n + 1) * n_samples]
        return xss

    @property
    def params(self) -> List[str]:
        return [*self.samples.param_names]


@dataclass
class Slice:
    """A slice of an MCMC trace."""

    lower: float
    upper: float


Domain = Tuple[float, float]  # normalised domain of a plot
Param = str
Slices = Dict[Param, List[Slice]]
Props = Dict[str, Any]
Point = Tuple[float, float]


# Bit inefficient for chains (recomputes min/max rather than used the cached property on ParameterSlices).
def normalise(xs: Sequence[float]) -> List[float]:
    """Normalise a sequence of numbers."""
    min_x: float = min(xs)
    max_x: float = max(xs)
    return [(x - min_x) / (max_x - min_x) for x in xs]


@dataclass
class ParameterSlices:
    """The MCMC data being presented."""

    slcs: List[Slice]
    param: str
    chains: np.ndarray  # shape is [n, n_iter] where n is number of chains
    max_sample: float
    min_sample: float

    @property
    def n_iter(self) -> int:
        """Return number of MCMC iterations per chain."""
        return int(self.chains.shape[1])

    def chain_slices(self, slc: Slice) -> List[np.ndarray]:
        """The specified slice of each chain."""
        return [
            self.chains[
                n,
                floor(slc.lower * self.n_iter):floor(slc.upper * self.n_iter)
            ]
            for n, _ in enumerate(self.chains)
        ]
