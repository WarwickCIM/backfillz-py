from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import sys
from typing import List

from stan.fit import Fit  # type: ignore

from backfillz.BackfillzTheme import BackfillzTheme, default, demo_1, demo_2, solarized_dark


class HistoryEvent(Enum):
    """A category of history event."""

    OBJECT_CREATION = 1


@dataclass
class HistoryEntry:
    """An entry in the Backfillz history log."""

    ident: int
    date: datetime
    event: HistoryEvent
    python_version: str
    saved: bool
    strings_as_factors: bool  # not sure what this is


class Backfillz:
    """Represents a Backfillz user session."""

    theme: BackfillzTheme
    plot_history: List[HistoryEntry]

    def __init__(self, fit: Fit) -> None:
        """Initialise a Backfillz session."""
        fit = fit
        self.set_theme("default", False)
        self.plot_history = [
            HistoryEntry(
                ident=1,
                date=datetime.now(),
                event=HistoryEvent.OBJECT_CREATION,
                python_version=sys.version,
                saved=False,
                strings_as_factors=False
            )
        ]

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


def as_backfillz(fit: Fit, verbose: bool) -> Backfillz:
    """Create a Backfillz session from a PyStan fit."""
    return Backfillz(fit)
