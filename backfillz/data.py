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
    start: float = min(xs)
    end: float = max(xs)
    return scale(1 / (end - start), translate(-start, xs))


@dataclass
class Axis:
    """Maps a range into a domain."""
    range: Domain
    domain: Domain

    def translate(self, x: float) -> float:
        r_start, r_end = self.range
        d_start, d_end = self.domain
        assert r_start <= x <= r_end
        return (x - r_start) / (r_end - r_start) * (d_end - d_start) + d_start


def normalise2(xs: Sequence[float], domain: Domain) -> Axis:
    return Axis((min(xs), max(xs)), domain)


def scale(factor: float, xs: Sequence[float]) -> List[float]:
    """Element-wise product."""
    return [x * factor for x in xs]


def translate(offset: float, xs: Sequence[float]) -> List[float]:
    """Element-wise addition of a constant."""
    return [x + offset for x in xs]


def segment(domain: Domain, n: int, m: int) -> Domain:
    """Break supplied "domain" into n equal-sized segments, and return the mth."""
    assert n > 0 and 0 <= m < n
    start, end = domain
    width = (end - start) / n
    return start + m * width, start + (m + 1) * width


def to_domain(x: float, domain: Domain) -> float:
    """Convert normalised coordinate to position within supplied domain."""
    start, end = domain
    return start + x * (end - start)


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
