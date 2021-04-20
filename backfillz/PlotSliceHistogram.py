from math import ceil, floor
from typing import List

from bokeh.models import LinearAxis  # type: ignore
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
    max_sample: float = np.amax(backfillz.mcmc_samples[param])
    min_sample: float = np.amin(backfillz.mcmc_samples[param])
    plot = {'parameter': param, 'sample_min': min_sample, 'sample_max': max_sample}
    print(plot)

    # Check, order and tag the slice
    param_col = slices['parameters']
    max_order: int = 0

    # TODO: ugh
    def count_param(param2: str) -> int:
        if param == param2:
            nonlocal max_order
            max_order += 1
            return max_order
        else:
            return 0  # R version puts NaN here, but maybe doesn't matter

    param_col2 = param_col.map(count_param)
    slices = pd.concat([slices, param_col2.to_frame('order')], axis=1)

    output_file("temp.html")
    fig: Figure = figure(
        title=f"Trace slice histogram of {param}",
        plot_width=400,
        plot_height=400,
        toolbar_location=None
    )
    fig.title.text_font_size = f"{backfillz.theme.text_cex_title}em"
    # TODO: set title colour to backfillz@theme.text_col_title
    fig.yaxis.minor_tick_line_color = None
    fig.xaxis.visible = False
    fig.xgrid.visible = False
    fig.ygrid.visible = False

    middle_width: int = 30  # check against R version
    height: int = n_iter

    # MIDDLE: JOINING SEGMENTS--------------------------------------
    slices.loc[param_col == param].apply(
        lambda slc: _create_slice(
            backfillz,
            fig,
            slc['lower'],
            slc['upper'],
            slc['order'],
            max_order=max_order,
            x_offset=max_sample,
            width=middle_width,
            y_scale=height
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

    x_axis = LinearAxis(bounds=(min_sample, max_sample))
    x_axis.minor_tick_line_color = None
    fig.add_layout(x_axis, 'below')

    # RIGHT: SLICE HISTOGRAM AND SAMPLE DENSITY ----------------------
    histogram_height: float = height / max_order
    slices.loc[param_col == param].apply(
        lambda slc: _create_slice_histogram(
            backfillz,
            fig,
            chains,
            slc['lower'],
            slc['upper'],
            min_sample=min_sample,
            max_sample=max_sample,
            x_start=max_sample + middle_width,
            height=histogram_height,
            y_start=(slc['order'] - 1) * histogram_height
        ),
        axis=1
    )

    show(fig)


def _create_slice(
    backfillz: Backfillz,
    fig: Figure,
    lower: float,
    upper: float,
    order: int,
    max_order: int,
    x_offset: float,
    width: int,
    y_scale: int
) -> None:
    fig.patch(
        _translate(x_offset, _scale(width, [0, 1, 1, 0])),
        _scale(y_scale, [lower, (order - 1) / max_order, order / max_order, upper]),
        color=backfillz.theme.bg_colour,
        alpha=0.5,
        line_width=1,
        # border=NA           TO DO
    )
    fig.line(
        _translate(x_offset, _scale(width, [0, 1])),
        _scale(y_scale, [lower, (order - 1) / max_order]),
        line_width=1,
        color=backfillz.theme.fg_colour
    )
    fig.line(
        _translate(x_offset, _scale(width, [0, 1])),
        _scale(y_scale, [upper, order / max_order]),
        line_width=1,
        color=backfillz.theme.fg_colour
    )


def _create_slice_histogram(
    backfillz: Backfillz,
    p: Figure,
    chains: np.ndarray,
    lower: float,
    upper: float,
    min_sample: float,
    max_sample: float,
    x_start: float,
    y_start: float,
    height: float
) -> None:
    x_start = x_start - min(min_sample, 0)
    [_, n] = chains.shape
    # first chain only for now; need to consider all?
    hist, edges = np.histogram(
        chains[0, floor(lower * n):floor(upper * n)],
        bins=np.linspace(start=floor(min_sample), stop=ceil(max_sample), num=40)
    )
    y_max = max(hist)
    p.quad(
        bottom=y_start,
        top=[y_start + y / y_max * height for y in hist],
        left=[x_start + x for x in edges[:-1]],
        right=[x_start + x for x in edges[1:]],
        fill_color=backfillz.theme.bg_colour,
        line_color=backfillz.theme.fg_colour
    )


def _scale(factor: float, xs: List[float]) -> List[float]:
    return [x * factor for x in xs]


def _translate(offset: float, xs: List[float]) -> List[float]:
    return [x + offset for x in xs]
