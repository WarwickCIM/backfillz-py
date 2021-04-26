from dataclasses import dataclass
from math import ceil, floor
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from backfillz.Backfillz import Backfillz, HistoryEntry, HistoryEvent
from backfillz.BackfillzTheme import BackfillzTheme


@dataclass
class Slice:
    """A slice of an MCMC trace."""

    lower: float
    upper: float


Slices = Dict[str, List[Slice]]


def plot_slice_histogram(backfillz: Backfillz, save_plot: bool = False) -> None:
    """Plot a slice histogram."""
    params = pd.Series(backfillz.mcmc_samples.param_names[0:1])  # just first param for now
    slice_list: List[Slice] = [
        Slice(0, 0.004), Slice(0.004, 0.01), Slice(0.01, 0.02), Slice(0.02, 0.04), Slice(0.04, 1)
    ]
    slices: Slices = {param: slice_list for param in params}

    for param in params:
        _create_single_plot(backfillz, slices[param], param)

    # Update log
    backfillz.plot_history.append(HistoryEntry(HistoryEvent.SLICE_HISTOGRAM, save_plot))


# Assume scalar parameter for now; what about vectors?
def _create_single_plot(
    backfillz: Backfillz,
    slices: List[Slice],
    param: str
) -> None:
    chains = backfillz.iter_chains(param)
    [n_chains, n_iter] = chains.shape
    print(f"iterations: {n_iter}, chains: {n_chains}, parameter: {param}")
    max_sample: float = np.amax(backfillz.mcmc_samples[param])
    min_sample: float = np.amin(backfillz.mcmc_samples[param])
    plot = dict(parameter=param, sample_min=min_sample, sample_max=max_sample)
    print(plot)

    middle_width: int = 30  # check against R version

    layout: go.Layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)

    fig: go.Figure = go.Figure(layout=layout)
    specs: List[List[object]] = \
        [[dict(rowspan=len(slices)), dict(rowspan=len(slices)), dict()]] + \
        [[None, None, dict()] for _ in slices[1:]]
    make_subplots(
        rows=len(slices),
        cols=3,
        figure=fig,
        specs=specs,
        horizontal_spacing=0,
        vertical_spacing=0,
        print_grid=True
    )

    # p.title=f"Trace slice histogram of {param}",
    # p.title.text_color = backfillz.theme.text_col_title

    # LEFT: TRACE PLOT ------------------------------------------
    trace_plots: Iterator[go.Scatter] = map(
        lambda n: (go.Scatter(
            x=chains[n],
            y=list(range(0, chains[n].size)),
            line=dict(color=backfillz.theme.palette[n])
        )),
        range(0, n_chains)
    )
    for trace_plot in trace_plots:
        fig.add_trace(trace_plot, row=1, col=1)
    fig.layout['yaxis'].update(range=[0, n_iter])

    # MIDDLE: JOINING SEGMENTS--------------------------------------
    for n_slice, slc in enumerate(slices, start=1):
        traces: List[go.Scatter] = _create_slice(
            backfillz,
            fig,
            slc,
            n_slice,
            max_order=len(slices),
            x_offset=max_sample,
            width=middle_width,
            y_scale=n_iter
        )
        for trace in traces:
            fig.add_trace(trace, row=1, col=2)
    fig.layout['yaxis2'].update(range=[0, n_iter])

    # RIGHT: SLICE HISTOGRAM AND SAMPLE DENSITY ----------------------
    for n_slice, slc in enumerate(slices):
        histo: go.Histogram = _slice_histogram(
            backfillz.theme,
            chains,
            slc,
            min_sample=min_sample,
            max_sample=max_sample
        )
        fig.add_trace(histo, row=len(slices) - n_slice, col=3)

    fig.show()


def _create_slice(
    backfillz: Backfillz,
    fig: go.Figure,
    slc: Slice,
    order: int,
    max_order: int,
    x_offset: float,
    width: int,
    y_scale: int
) -> List[go.Scatter]:
    traces: List[go.Scatter] = [
        go.Scatter(
            x=_translate(x_offset, _scale(width, [0, 1, 1, 0])),
            y=_scale(y_scale, [slc.lower, (order - 1) / max_order, order / max_order, slc.upper]),
            mode='lines',
            line=dict(width=0),
            fill='toself',
            fillcolor='rgba(240,240,240,255)'
        ),
        go.Scatter(
            x=_translate(x_offset, _scale(width, [0, 1])),
            y=_scale(y_scale, [slc.lower, (order - 1) / max_order]),
            mode='lines',
            line=dict(color=backfillz.theme.fg_colour, width=1)
        ),
        go.Scatter(
            x=_translate(x_offset, _scale(width, [0, 1])),
            y=_scale(y_scale, [slc.upper, order / max_order]),
            mode='lines',
            line=dict(color=backfillz.theme.fg_colour, width=1)
        ),
    ]

    return traces


def _slice_histogram(
    theme: BackfillzTheme,
    chains: np.ndarray,
    slc: Slice,
    min_sample: float,
    max_sample: float
) -> go.Histogram:
    [_, n] = chains.shape
    # chain 0 only for now; need to consider all?
    return go.Histogram(
        x=chains[0, floor(slc.lower * n):floor(slc.upper * n)],
        xbins=dict(start=floor(min_sample), end=ceil(max_sample), size=1),
        marker=dict(color=theme.bg_colour, line=dict(color=theme.fg_colour, width=1))
    )


def _scale(factor: float, xs: List[float]) -> List[float]:
    return [x * factor for x in xs]


def _translate(offset: float, xs: List[float]) -> List[float]:
    return [x + offset for x in xs]
