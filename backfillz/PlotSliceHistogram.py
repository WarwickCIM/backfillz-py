from typing import List

from bokeh.plotting import Figure, figure, output_file, show  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from backfillz.Backfillz import Backfillz, HistoryEntry, HistoryEvent


def plot_slice_histogram(backfillz: Backfillz, save_plot: bool = False) -> None:
    """Plot a slice histogram."""
    params = pd.Series(backfillz.mcmc_samples.param_names[0:2])
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
    [dims, n_draws] = backfillz.mcmc_samples[param].shape
    print(f"draws: {n_draws}, dims: {dims}, parameter: {param}")
    max_sample = np.amax(backfillz.mcmc_samples[param])
    min_sample = np.amin(backfillz.mcmc_samples[param])
    plot = {'parameter': param, 'sample_min': min_sample, 'sample_max': max_sample}
    print(f'Creating plot for { plot }')

    # Check, order and tag the slice
    param_col = slices['parameters']
    param_count: int = 0

    # TODO: better idiom
    def count_param(param2: str) -> int:
        if param == param2:
            nonlocal param_count
            param_count += 1
            return param_count
        else:
            return pd.NA

    param_col2 = param_col.map(count_param)
    slices = pd.concat([slices, param_col2.to_frame('order')], axis=1)

    output_file("temp.html")
    fig: Figure = figure(plot_width=400, plot_height=400)

    slices.loc[param_col == param].apply(
        lambda row: _create_slice(backfillz, fig, row['lower'], row['upper'], row['order'], param_count),
        axis=1
    )

    # line_plot <- function(x) {
    #   lines(x[-1],
    #         1:n,
    #         col = alpha(object@theme_palette[[x[1]]], 1),
    #   )
    # }

    # # Plot every chain
    xs = backfillz.mcmc_samples[param][0]  # scalar parameter; what about vectors?
    fig.line(
        xs,
        range(0, xs.size),
        line_width=1,
        color="black"
    )

    show(fig)


# y not used..?
def _create_slice(backfillz: Backfillz, fig: Figure, lower: float, upper: float, order: int, max_order: int) -> None:
    print(lower, upper, order, max_order)
    fig.patch(
        [0, 1, 1, 0],
        [lower, (order - 1) / max_order, order / max_order, upper],
        color="gray",  # backfillz.theme.bg_colour,
        alpha=0.5,
        line_width=1,
        # border=NA           TO DO
    )
    fig.line(
        [0, 1],
        [lower, (order - 1) / max_order],
        line_width=2,
        color="red"  # backfillz.theme.fg_colour
    )
    fig.line(
        [0, 1],
        [upper, order / max_order],
        line_width=2,
        color="blue"  # backfillz.theme.fg_colour
    )

