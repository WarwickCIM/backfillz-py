from dataclasses import dataclass
from math import floor
from typing import Any, Dict, List, Tuple

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
        return list(self.samples.param_names)


Domain = Tuple[float, float]  # normalised domain of a plot
Param = str
Slices = Dict[Param, List[Domain]]
Props = Dict[str, Any]
Point = Tuple[float, float]


def size(domain: Domain) -> float:
    """Size of a domain."""
    lower, upper = domain
    return upper - lower


@dataclass
class Axis:
    """Map a range into a domain."""

    range: Domain
    domain: Domain

    def map(self, x: float) -> float:
        start, end = self.range
        return to_domain((x - start) / (end - start), self.domain)


def axis(xs: List[float], x_domain: Domain) -> Axis:
    """Map range of numbers to a domain."""
    return Axis((min(xs), max(xs)), x_domain)


def to_domain(x: float, domain: Domain) -> float:
    """Convert normalised x coordinate to coordinate within supplied angular domain."""
    start, end = domain
    return start + x * (end - start)


def map_domain(range: Domain, domain: Domain) -> Domain:
    """Map a domain into another."""
    x_axis: Axis = Axis(range, domain)
    start, end = range
    return x_axis.map(start), x_axis.map(end)


def normalise(xs: List[float]) -> List[float]:
    """Normalise a list of floats."""
    x_axis: Axis = axis(xs, (0, 1))
    return [x_axis.map(x) for x in xs]


def scale(factor: float, xs: List[float]) -> List[float]:
    """Element-wise multiplication by a constant."""
    return [x * factor for x in xs]


def segment(domain: Domain, n: int, m: int) -> Domain:
    """Break supplied "domain" into n equal-sized segments, and return the mth."""
    start, end = domain
    width = (end - start) / n
    return start + m * width, start + (m + 1) * width


@dataclass
class ParameterSlices:
    """The MCMC data being presented."""

    slcs: List[Domain]
    param: str
    chains: np.ndarray  # shape is [n, n_iter] where n is number of chains
    max_sample: float
    min_sample: float

    @property
    def n_iter(self) -> int:
        """Return number of MCMC iterations per chain."""
        return int(self.chains.shape[1])

    def chain_slices(self, slc: Domain) -> List[np.ndarray]:
        """The specified slice of each chain."""
        lower, upper = slc
        return [
            self.chains[n, floor(lower * self.n_iter):floor(upper * self.n_iter)]
            for n, _ in enumerate(self.chains)
        ]
