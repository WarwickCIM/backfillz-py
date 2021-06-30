from dataclasses import dataclass
from typing import List

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
