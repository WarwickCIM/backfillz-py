from datetime import datetime
from enum import Enum
import sys
from typing import List

import numpy as np
from stan.fit import Fit  # type: ignore

from backfillz.BackfillzTheme import BackfillzTheme, default, demo_1, demo_2, solarized_dark


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
        self.set_theme("default", False)
        self.plot_history = [
            HistoryEntry(HistoryEvent.OBJECT_CREATION, False)
        ]

    def iter_chains(self, param: str, index: int = 0) -> np.ndarray:
        """Return the (num_chains x num_samples) matrix of draws for a given parameter."""
        num_chains = self.mcmc_samples.num_chains
        num_samples = self.mcmc_samples.num_samples
        xss = np.zeros((num_chains, num_samples))
        print(xss)
        for n in range(0, num_chains):
            xss[n] = self.mcmc_samples[param][index][n * num_samples: (n + 1) * num_samples]
        return xss

    def set_theme(self, theme: str, verbose: bool = True) -> None:
        """Set Backfillz theme."""
        if verbose:
            print("Setting backfillz object theme to " + theme)
        if theme == "default":
            self.theme = default
        elif theme == "solarized_dark":
            self.theme = solarized_dark
        elif theme == "demo 1":
            self.theme = demo_1
        elif theme == "demo 2":
            self.theme = demo_2
        else:
            raise Exception("Theme not recognised")
