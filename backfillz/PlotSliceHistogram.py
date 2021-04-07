import numpy as np
import pandas as pd  # type: ignore

from backfillz.Backfillz import Backfillz, HistoryEntry, HistoryEvent


def plot_slice_histogram(backfillz: Backfillz, save_plot: bool = False) -> None:
    """Plot a slice histogram."""
    params = pd.Series(backfillz.fit.param_names[0:2])
    lower = pd.Series([0, 0.8])
    upper = pd.Series([0.4, 1])
    slices: pd.DataFrame = pd.DataFrame(columns=[
        'parameters',  # character
        'lower',  # numeric
        'upper'  # numeric
    ])
    for param in params:
        slices = pd.concat([
            slices,
            pd.DataFrame({
                'parameters': pd.Series([param] * upper.size),
                'lower': lower,
                'upper': upper
            }),
        ], ignore_index=True)

    for param in params:
        _create_single_plot(backfillz, slices, param)

    # Update log
    backfillz.plot_history.append(HistoryEntry(HistoryEvent.SLICE_HISTOGRAM, save_plot))


def _create_single_plot(backfillz: Backfillz, slices: pd.DataFrame, param: str) -> None:
    print(backfillz.fit[param].shape)
    max_sample = np.amax(backfillz.fit[param])
    min_sample = np.amin(backfillz.fit[param])
    plot = {'parameter': param, 'sample_min': min_sample, 'sample_max': max_sample}
    print(f'Creating plot for { plot }')
