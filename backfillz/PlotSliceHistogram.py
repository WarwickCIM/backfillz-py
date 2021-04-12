import numpy as np
import pandas as pd  # type: ignore
from bokeh.plotting import figure, Figure, show

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
    [n_chains, n] = backfillz.fit[param].shape
    print(slices)
    max_sample = np.amax(backfillz.fit[param])
    min_sample = np.amin(backfillz.fit[param])
    plot = {'parameter': param, 'sample_min': min_sample, 'sample_max': max_sample}
    print(f'Creating plot for { plot }')

    # Check, order and tag the slice
    param_col = slices['parameters']
    print(slices.loc[param_col == param])
    param_col2 = param_col.map(lambda param2: slices.loc[param_col == param].shape[0] if param == param2 else pd.NA)
    # concat is pure in Python, but not sure if we need imperative update anyway
    slices = pd.concat([slices, param_col2], axis=1)

    fig: Figure = figure(plot_width=400, plot_height=400)

    # Graphics parameters to find Python equivalent of:

    # par(fig = c(0.08 + 1 / 3, 2 / 3 - 0.08, 0.25, 0.85),
    #    family = object@theme_text_family,
    #    font = object@theme_text_font,
    #    bg = object@theme_bg_colour,
    #    fg = object@theme_fg_colour,
    #    col.lab = object@theme_text_font_colour,
    #    col.axis = object@theme_text_font_colour,
    #    cex.axis = object@theme_text_cex_axis,
    #    cex.main = object@theme_text_cex_title,
    #    cex.lab = object@theme_text_cex_main,
    #    bty = "n")
    #
    # par(mar = c(0, 0, 0, 0))

    # Other preliminaries to do:
    # plot(
    #  0:1, 0:1, type = "n", yaxs = "i", axes = FALSE, xaxs = "i", ann = FALSE
    # )
    #
    # background rectangle - colour to match the rects in the Left Hand Plot
    # rect(0, 0, 1, 1, border = FALSE,
    #     col = adjustcolor(object@theme_mg_colour,
    #                       alpha.f = object@theme_alpha)
    # )


def create_slice(backfillz: Backfillz, fig: Figure, order, x, y) -> None:
    fig.patch(
        [0, 1, 1, 0],
        [x[1], (x[3] - 1) / max(order), x[3] / max(order), x[2]],
        color=backfillz.theme.bg_colour,
        alpha=0.5,
        line_width=1,
        # border=NA           TO DO
    )
    fig.line(
        [0, 1],
        [x[1], (x[3] - 1) / max(order)],
        color=backfillz.theme.fg_colour
    )
    fig.line(
        [0, 1],
        [x[2], x[3] / max(order)],
        color=backfillz.theme.fg_colour
    )
