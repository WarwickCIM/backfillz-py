from dataclasses import dataclass
from math import ceil, floor
from typing import List

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from backfillz.Backfillz import Backfillz, HistoryEntry, HistoryEvent
from backfillz.BackfillzTheme import BackfillzTheme


@dataclass
class Slice:
    lower: float
    upper: float


Slices = dict[str, List[Slice]]


def plot_slice_histogram(backfillz: Backfillz, save_plot: bool = False) -> None:
    """Plot a slice histogram."""
    params = pd.Series(backfillz.mcmc_samples.param_names[0:1])  # just first param for now
    slices2: Slices = {param: [Slice(0, 0.4), Slice(0.8, 1)] for param in params}
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
            pd.DataFrame(dict(
                parameters=pd.Series([param] * upper.size),
                lower=lower,
                upper=upper
            )),
        ], ignore_index=True)

    for param in params:
        _create_single_plot(backfillz, slices, slices2[param], param)

    # Update log
    backfillz.plot_history.append(HistoryEntry(HistoryEvent.SLICE_HISTOGRAM, save_plot))


# Assume scalar parameter for now; what about vectors?
def _create_single_plot(backfillz: Backfillz, slices: pd.DataFrame, slices2: List[Slice], param: str) -> None:
    chains = backfillz.iter_chains(param)
    [n_chains, n_iter] = chains.shape
    print(f"iterations: {n_iter}, chains: {n_chains}, parameter: {param}")
    max_sample: float = np.amax(backfillz.mcmc_samples[param])
    min_sample: float = np.amin(backfillz.mcmc_samples[param])
    plot = dict(parameter=param, sample_min=min_sample, sample_max=max_sample)
    print(plot)

    # Check, order and tag the slice
    param_col = slices['parameters']
    n_slices: int = 0

    # ugh -- do something about this
    def count_param(param2: str) -> int:
        if param == param2:
            nonlocal n_slices
            n_slices += 1
            return n_slices
        else:
            return 0  # R version puts NaN here, but maybe doesn't matter

    param_col2 = param_col.map(count_param)
    slices = pd.concat([slices, param_col2.to_frame('order')], axis=1)

    plot_height: int = 600
    middle_width: int = 30  # check against R version
    right_width: int = 300

    fig: go.Figure = go.Figure(
        layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    )
    specs: List[List[object]] = \
        [[dict(rowspan=n_slices), dict(rowspan=n_slices), dict()]] + \
        [[None, None, dict()] for _ in range(1, n_slices)]
    print(specs)
    make_subplots(
        rows=n_slices,
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
    for n in range(0, n_chains):
        fig.add_trace(go.Scatter(x=chains[n], y=list(range(0, chains[n].size))), row=1, col=1)

    # MIDDLE: JOINING SEGMENTS--------------------------------------
    slices2 = zip(slices2, range(0, len(slices2)))
    map(
        lambda slc_n: _create_slice(
            backfillz,
            fig,
            slc_n[0],
            slc_n[1],
            max_order=n_slices,
            x_offset=max_sample,
            width=middle_width,
            y_scale=n_iter
        ),
        slices2
    )

    # RIGHT: SLICE HISTOGRAM AND SAMPLE DENSITY ----------------------
    slices.loc[param_col == param].apply(
        lambda slc: _slice_histogram(
            backfillz.theme,
            fig,
            chains,
            Slice(slc['lower'], slc['upper']),
            slc['order'],
            min_sample=min_sample,
            max_sample=max_sample,
            width=right_width,
            height=(1 / n_slices) * plot_height
        ),
        axis=1
    )

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
) -> None:
    fig.add_trace(go.Scatter(
        x=_translate(x_offset, _scale(width, [0, 1, 1, 0])),
        y=_scale(y_scale, [slc.lower, (order - 1) / max_order, order / max_order, slc.upper]),
        mode='lines',
        line=dict(width=0),
        fill='toself',
        fillcolor='rgba(240,240,240,255)'
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=_translate(x_offset, _scale(width, [0, 1])),
        y=_scale(y_scale, [slc.lower, (order - 1) / max_order]),
        mode='lines',
        line=dict(color=backfillz.theme.fg_colour, width=1)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=_translate(x_offset, _scale(width, [0, 1])),
        y=_scale(y_scale, [slc.upper, order / max_order]),
        mode='lines',
        line=dict(color=backfillz.theme.fg_colour, width=1)
    ), row=1, col=2)


def _slice_histogram(
    theme: BackfillzTheme,
    fig: go.Figure,
    chains: np.ndarray,
    slc: Slice,
    slice_index: int,
    min_sample: float,
    max_sample: float,
    width: float,
    height: float
) -> None:
    [_, n] = chains.shape
    # chain 0 only for now; need to consider all?
    fig.add_trace(
        go.Histogram(
            x=chains[0, floor(slc.lower * n):floor(slc.upper * n)],
            xbins=dict(start=floor(min_sample), end=ceil(max_sample), size=1),
            marker=dict(color=theme.bg_colour, line=dict(color=theme.fg_colour, width=1))
        ),
        row=slice_index,
        col=3
    )


def _scale(factor: float, xs: List[float]) -> List[float]:
    return [x * factor for x in xs]


def _translate(offset: float, xs: List[float]) -> List[float]:
    return [x + offset for x in xs]
