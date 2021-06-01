from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import sys
from typing import Any, Dict, List

import numpy as np
from stan.fit import Fit  # type: ignore

from backfillz.theme import BackfillzTheme, default


class HistoryEvent(Enum):
    """Category of event."""

    OBJECT_CREATION = 1
    SLICE_HISTOGRAM = 2


class HistoryEntry:
    """An entry in the Backfillz history log."""

    count = 0

    ident: int
    date: datetime
    event: HistoryEvent
    python_version: str
    saved: bool

    def __init__(
        self,
        event: HistoryEvent,
        saved: bool
    ) -> None:
        """Construct a history entry."""
        self.ident = HistoryEntry.count
        HistoryEntry.count += 1
        self.date = datetime.now()
        self.event = event
        self.python_version = sys.version
        self.saved = saved


class Backfillz:
    """Represents a Backfillz user session."""

    theme: BackfillzTheme
    plot_history: List[HistoryEntry]

    def __init__(self, fit: Fit) -> None:
        """Initialise a Backfillz session."""
        self.mcmc_samples = fit
        self.set_theme(default, False)
        self.plot_history = [
            HistoryEntry(HistoryEvent.OBJECT_CREATION, False)
        ]

    def iter_chains(self, param: str, index: int = 0) -> np.ndarray:
        """Return the (n_chains x n_samples) matrix of draws for a given parameter."""
        n_chains, n_samples = self.mcmc_samples.num_chains, self.mcmc_samples.num_samples
        xss = np.zeros((n_chains, n_samples))
        for n in range(0, n_chains):
            xss[n] = self.mcmc_samples[param][index][n * n_samples: (n + 1) * n_samples]
        return xss

    def set_theme(self, theme: BackfillzTheme, verbose: bool = True) -> None:
        """Set Backfillz theme."""
        if verbose:
            print("Setting backfillz object theme to " + theme.name)
        self.theme = theme


@dataclass
class Slice:
    """A slice of an MCMC trace."""

    lower: float
    upper: float


Param = str
Slices = Dict[Param, List[Slice]]
Props = Dict[str, Any]


@dataclass
class ParameterSlices:
    """The MCMC data being presented."""

    slcs: List[Slice]
    param: str
    chains: np.ndarray
    max_sample: float
    min_sample: float

    @property
    def n_chains(self) -> int:
        return int(self.chains.shape[0])

    @property
    def n_iter(self) -> int:
        """Return number of MCMC iterations per chain."""
        return int(self.chains.shape[1])

    @property
    def n_slcs(self) -> int:
        return len(self.slcs)
