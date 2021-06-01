from abc import abstractmethod
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
AxisId = Optional[int]
AxisIds = Tuple[AxisId, AxisId]
Props = Dict[str, Any]


def scale(factor: float, xs: List[float]) -> List[float]:
    """Element-wise product."""
    return [x * factor for x in xs]


def segment(domain: Tuple[float, float], n: int, m: int) -> Tuple[float, float]:
    """Break supplied "domain" into n equal-sized segments, and return the mth."""
    start, end = domain
    width = (end - start) / n
    return start + m * width, start + (m + 1) * width


@dataclass
class ParameterSlices:
    """The MCMC data being presented."""

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

    @property
    def n_slcs(self) -> int:
        return len(self.slcs)


@dataclass
class Plot:
    """Base class providing common subplot functionality."""

    axis_ids: List[AxisId]
    x_domain: Tuple[float, float]  # left/right edges normalised to [0, 1]
    y_domain: Tuple[float, float]  # top/bottom edges normalised to [0, 1]
    row: int
    col: int
    data: ParameterSlices
    theme: BackfillzTheme

    def layout_axes(self, fig: go.Figure) -> None:
        pass

    def render(self, fig: go.Figure) -> None:
        """Render me into fig."""
        pass

    @property
    def top_left(self) -> Tuple[float, float]:
        return self.x_domain[0], self.y_domain[1]


@dataclass
class Subplot(Plot):
    """A leaf subplot."""

    @property
    def axis_defaults(self) -> Dict[str, Any]:
        return dict(
            showgrid=False,
            zeroline=False,
            linecolor=self.theme.fg_colour,
            ticks='outside',
            tickwidth=1,
            ticklen=5,
            tickcolor=self.theme.fg_colour,
            fixedrange=True,  # disable selection zoom
        )

    @property
    def xaxis_id(self) -> str:
        """My Plotly-assigned x-axis id."""
        xaxis_id = self.axis_ids[0]
        return 'xaxis' + ('' if xaxis_id is None else str(xaxis_id))

    @property
    def yaxis_id(self) -> str:
        """My Plotly-assigned y-axis id."""
        yaxis_id = self.axis_ids[0]
        return 'yaxis' + ('' if yaxis_id is None else str(yaxis_id))

    def layout_axes(self, fig: go.Figure) -> None:
        """Configure my x and y axis settings in fig."""
        fig.layout[self.xaxis_id].update(domain=self.x_domain, **self.axis_defaults, **self.xaxis_props)
        fig.layout[self.yaxis_id].update(domain=self.y_domain, **self.axis_defaults, **self.yaxis_props)

    @property
    def xaxis_props(self) -> Props:
        """My custom x-axis settings; subclasses can override."""
        return dict()

    @property
    def yaxis_props(self) -> Props:
        """My custom y-axis settings; subclasses can override."""
        return dict()


@dataclass
class VerticalSubplots(Plot):
    """A collection of vertically arranged subplots."""

    _plots: Optional[List[Plot]] = None

    # Want @cached_property but Mypy doesn't seem to support it properly.
    @property
    def plots(self) -> List[Plot]:
        if self._plots is None:
            self._plots = self.make_plots()
        return self._plots

    @abstractmethod
    def make_plots(self) -> List[Plot]:
        """My subplots."""
        pass

    def layout_axes(self, fig: go.Figure) -> None:
        """Ask each subplot to configure its axes."""
        for plot in self.plots:
            plot.layout_axes(fig)

    def render(self, fig: go.Figure) -> None:
        """Render subplots into fig."""
        for n, plot in enumerate(self.plots):
            plot.render(fig)
