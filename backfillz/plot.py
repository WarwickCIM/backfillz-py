from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go  # type: ignore

from backfillz.theme import BackfillzTheme


@dataclass
class Slice:
    """A slice of an MCMC trace."""

    lower: float
    upper: float


Param = str
Slices = Dict[Param, List[Slice]]

# ints assigned as axis id suffixes by Plotly; omitted for first subplot
AxisIds = Tuple[Optional[int], Optional[int]]
Props = Dict[str, Any]


def nth_axes_of(axis_ids: AxisIds, n: int, count: int) -> AxisIds:
    """For non-None axes, nth (from 0) pair of axis ids counting *up* (where axis ids grown *down*)."""
    [xaxis_id, yaxis_id] = axis_ids
    assert isinstance(xaxis_id, int) and isinstance(yaxis_id, int)
    return xaxis_id + count - 1 - n, yaxis_id + count - 1 - n


def _scale(factor: float, xs: List[float]) -> List[float]:
    return [x * factor for x in xs]


def segment(domain: Tuple[float, float], n: int, m: int) -> Tuple[float, float]:
    """Break supplied "domain" into n equal-sized segments, and return the mth."""
    start, end = domain
    width = (end - start) / n
    return start + m * width, start + (m + 1) * width


@dataclass
class ChartData:
    """The MCMC data being presented."""

    theme: BackfillzTheme
    slcs: List[Slice]
    param: str
    chains: np.ndarray
    max_sample: float
    min_sample: float

    @property
    def n_chains(self) -> int:
        return int(self.chains.shape[0])

    @property
    def n_iter(self) -> int:
        """Return number of MCMC iterations per chain."""
        return int(self.chains.shape[1])


@dataclass
class Plot:
    """Base class of Subplot and Subplots."""

    axis_ids: AxisIds
    y_domain: Tuple[float, float]
    data: ChartData

    @property
    def xaxis_id(self) -> str:
        """My Plotly-assigned x-axis id; for an aggregate, my first such id."""
        xaxis_id = self.axis_ids[0]
        return 'xaxis' + ('' if xaxis_id is None else str(xaxis_id))

    @property
    def yaxis_id(self) -> str:
        """My Plotly-assigned y-axis id; for an aggregate, my first such id."""
        yaxis_id = self.axis_ids[1]
        return 'yaxis' + ('' if yaxis_id is None else str(yaxis_id))

    def layout_axes(self, fig: go.Figure) -> None:
        pass

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        """Render me into fig at supplied row and column."""
        pass


@dataclass
class Subplot(Plot):
    """A Plotly subplot and its assigned axis ids."""

    def layout_axes(self, fig: go.Figure) -> None:
        """Configure my x and y axis settings in fig."""
        fig.layout[self.xaxis_id].update(**self.xaxis_props)
        fig.layout[self.yaxis_id].update(**self.yaxis_props)

    @property
    def xaxis_props(self) -> Props:
        """My custom x-axis settings; subclasses can override."""
        return dict()

    @property
    def yaxis_props(self) -> Props:
        """My custom y-axis settings; subclasses can override."""
        return dict()


@dataclass
class Subplots(Plot):
    """A collection of vertically arranged subplots."""

    # Unfortunately mypy doesn't support @cached-property properly
    def layout_axes(self, fig: go.Figure) -> None:
        """Ask each subplot to configure its axes."""
        for plot in self.plots:  # type: ignore[attr-defined]
            plot.layout_axes(fig)

    def render(self, fig: go.Figure, row: int, col: int) -> None:
        """Render subplots into fig, placing subplots into ascending (from 0) rows starting from bottom."""
        for n, plot in enumerate(self.plots):  # type: ignore[attr-defined]
            plot.render(fig, row=row + len(self.plots) - 1 - n, col=col)
