from datetime import datetime
import sys

import pandas as pd  # type: ignore

from backfillz.Backfillz import Backfillz, HistoryEntry, HistoryEvent


def plot_slice_histogram(backfillz: Backfillz, save_plot: bool = False) -> None:
    """Plot a slice histogram."""
    pd.DataFrame(columns=[  # TODO: store in Backfillz object
        'parameter',  # character
        'sample_min',  # numeric
        'sample_max'  # numeric
    ])

    # array(attributes(backfillz.fit).dimnames.parameters)[1:2]
    parameters = pd.Series(['a', 'b'])  # todo
    lower = pd.Series([0, 0.8])
    upper = pd.Series([0.4, 1])
    slices: pd.DataFrame = pd.DataFrame(columns=[
        'parameters',  # character
        'lower',  # numeric
        'upper'  # numeric
    ])
    for parameter in parameters:
        slices = pd.concat([
            slices,
            pd.DataFrame({
                'parameters': pd.Series([parameter] * upper.size),
                'lower': lower,
                'upper': upper
            }),
        ], ignore_index=True)

    for parameter in parameters:
        _create_single_plot(slices, parameter)

    ident = max(map(lambda entry: entry.ident, backfillz.plot_history)) + 1

    # Update log
    backfillz.plot_history.append(HistoryEntry(HistoryEvent.SLICE_HISTOGRAM, save_plot))


def _create_single_plot(slices: pd.DataFrame, parameter: str) -> None:
    pass
