from math import ceil, floor
from typing import List

from bokeh.layouts import column, row  # type: ignore
from bokeh.models import LinearAxis, Range1d  # type: ignore
from bokeh.plotting import Figure, figure, output_file, show
import numpy as np
import pandas as pd  # type: ignore

from backfillz.Backfillz import Backfillz, HistoryEntry, HistoryEvent


def plot_slice_histogram(backfillz: Backfillz, save_plot: bool = False) -> None:
    """Plot a slice histogram."""
    params = pd.Series(backfillz.mcmc_samples.param_names[0:1])  # just first param for now
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

    plot_height: int = 600
    output_file("temp.html")
    p: Figure = figure(
#        title=f"Trace slice histogram of {param}",
        plot_width=800,
        plot_height=plot_height,
        toolbar_location=None
    )
    p.min_border = 2
    p.title.text_font_size = f"{backfillz.theme.text_cex_title}em"
    # TODO: set title colour to backfillz@theme.text_col_title
    p.yaxis.minor_tick_line_color = None
    p.yaxis.fixed_location = min_sample
    p.yaxis.bounds = (0, n_iter)
    p.xaxis.visible = False
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.outline_line_color = None

    # LEFT: TRACE PLOT ------------------------------------------
    for n in range(0, n_chains):
        p.line(
            chains[n],
            range(0, chains[n].size),
            line_width=1,
            color=backfillz.theme.palette[n]
        )

    xaxis = LinearAxis(bounds=(min_sample, max_sample))
    xaxis.minor_tick_line_color = None
    xaxis.fixed_location = 0
    p.add_layout(xaxis, 'below')

    middle_width: int = 30  # check against R version

    # MIDDLE: JOINING SEGMENTS--------------------------------------
    slices.loc[param_col == param].apply(
        lambda slc: _create_slice(
            backfillz,
            p,
            slc['lower'],
            slc['upper'],
            slc['order'],
            max_order=max_order,
            x_offset=max_sample,
            width=middle_width,
            y_scale=n_iter
        ),
        axis=1
    )

    # RIGHT: SLICE HISTOGRAM AND SAMPLE DENSITY ----------------------
    hgrams = slices.loc[param_col == param].apply(
        lambda slc: _slice_histogram(
            backfillz,
            chains,
            slc['lower'],
            slc['upper'],
            min_sample=min_sample,
            max_sample=max_sample,
            height=(1 / max_order) * plot_height
        ),
        axis=1
    )

    show(row(p, column(hgrams.tolist())))


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


def _slice_histogram(
    backfillz: Backfillz,
    chains: np.ndarray,
    lower: float,
    upper: float,
    min_sample: float,
    max_sample: float,
    height: float
) -> Figure:
    [_, n] = chains.shape
    x_start = -min(min_sample, 0)
    # first chain only for now; need to consider all?
    hist, edges = np.histogram(
        chains[0, floor(lower * n):floor(upper * n)],
        bins=np.linspace(start=floor(min_sample), stop=ceil(max_sample), num=40)
    )
    y_max = max(hist)
    print(y_max)

    p = figure(plot_width=200, plot_height=int(height), toolbar_location=None, y_axis_location='right')
    p.min_border = 1  # else bottom edge of each bar is clipped
    p.x_range = Range1d(min_sample, max_sample)
    p.y_range = Range1d(0, y_max)
    p.xaxis.visible = False
    p.yaxis.minor_tick_line_color = None
    p.yaxis.bounds = (0, n)
    p.grid.visible = False
    p.outline_line_color = None
    p.quad(
        bottom=0,
        top=hist,
        left=[x_start + x for x in edges[:-1]],
        right=[x_start + x for x in edges[1:]],
        fill_color=backfillz.theme.bg_colour,
        line_color=backfillz.theme.fg_colour
    )
    return p

def _scale(factor: float, xs: List[float]) -> List[float]:
    return [x * factor for x in xs]


def _translate(offset: float, xs: List[float]) -> List[float]:
    return [x + offset for x in xs]
