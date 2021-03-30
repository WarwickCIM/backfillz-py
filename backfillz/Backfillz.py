import pandas as pd
from stan.fit import Fit  # type: ignore

from backfillz.BackfillzTheme import BackfillzTheme, default, demo_1, demo_2, solarized_dark


class Backfillz:
    """Represents a Backfillz user session."""

    fit: Fit
    theme: BackfillzTheme
    plot_history: pd.DataFrame

    def __init__(self, mcmc_samples: Fit) -> None:
        """Initialise a Backfillz session."""
        #  backfillz_object@plot_history <- data.frame(
        #    ID = 1,
        #    Date = date(),
        #    Event = "Object Creation",
        #    R_version = R.Version()$version.string,
        #    Saved = FALSE,
        #    stringsAsFactors = FALSE
        #  )
        pass

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
    backfillz = Backfillz(fit)
    backfillz.set_theme("default", verbose)
    return backfillz
