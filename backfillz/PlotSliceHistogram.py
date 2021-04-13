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


# Assume scalar parameter for now; what about vectors?
def _create_single_plot(backfillz: Backfillz, slices: pd.DataFrame, param: str) -> None:
    chains = backfillz.iter_chains(param)
    [n_chains, n_iter] = chains.shape
    print(f"iterations: {n_iter}, chains: {n_chains}, parameter: {param}")
    max_sample = np.amax(backfillz.mcmc_samples[param])
    min_sample = np.amin(backfillz.mcmc_samples[param])
    plot = {'parameter': param, 'sample_min': min_sample, 'sample_max': max_sample}
    print(plot)

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
            return 0  # R version puts NaN here, but maybe doesn't matter

    param_col2 = param_col.map(count_param)
    slices = pd.concat([slices, param_col2.to_frame('order')], axis=1)

    output_file("temp.html")
    fig: Figure = figure(plot_width=400, plot_height=400)

    # MIDDLE: JOINING SEGMENTS--------------------------------------
    slices.loc[param_col == param].apply(
        lambda row: _create_slice(
            backfillz,
            fig,
            row['lower'],
            row['upper'],
            row['order'],
            param_count,
            30,  # hard-coded for now
            n_iter
        ),
        axis=1
    )

    # LEFT: TRACE PLOT ------------------------------------------
    for n in range(0, n_chains):
        fig.line(
            chains[n],
            range(0, chains[n].size),
            line_width=1,
            color=backfillz.theme.palette[n]
        )

    show(fig)


def _create_slice(
    backfillz: Backfillz,
    fig: Figure,
    lower: float,
    upper: float,
    order: int,
    max_order: int,
    x_scale: int,
    y_scale: int
) -> None:
    fig.patch(
        _scale(x_scale, [0, 1, 1, 0]),
        _scale(y_scale, [lower, (order - 1) / max_order, order / max_order, upper]),
        color="gray",  # backfillz.theme.bg_colour,
        alpha=0.5,
        line_width=1,
        # border=NA           TO DO
    )
    fig.line(
        _scale(x_scale, [0, 1]),
        _scale(y_scale, [lower, (order - 1) / max_order]),
        line_width=2,
        color="red"  # backfillz.theme.fg_colour
    )
    fig.line(
        _scale(x_scale, [0, 1]),
        _scale(y_scale, [upper, order / max_order]),
        line_width=2,
        color="blue"  # backfillz.theme.fg_colour
    )


def _scale(factor: float, xs: List[float]) -> List[float]:
    return [x * factor for x in xs]
