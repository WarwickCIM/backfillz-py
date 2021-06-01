from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import plotly.graph_objects as go  # type: ignore

from backfillz.core import ParameterSlices, Props
from backfillz.theme import BackfillzTheme


# ints assigned as axis id suffixes by Plotly; omitted for first subplot
AxisId = Optional[int]


def scale(factor: float, xs: List[float]) -> List[float]:
    """Element-wise product."""
    return [x * factor for x in xs]


def segment(domain: Tuple[float, float], n: int, m: int) -> Tuple[float, float]:
    """Break supplied "domain" into n equal-sized segments, and return the mth."""
    start, end = domain
    width = (end - start) / n
    return start + m * width, start + (m + 1) * width


def annotate(
    fig: go.Figure,
    font_size: int,
    at: Tuple[float, float],
    xanchor: Literal['left', 'right'],
    yanchor: Literal['top', 'bottom'],
    y_adjust: Optional[float],  # additional normalised offet of text relative to plot
    text: str,
) -> None:
    """Add an annotation to supplied figure, with supplied arguments in addition to some default settings."""
    fig.add_annotation(
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(size=font_size),
        x=at[0],
        y=at[1] + (0 if y_adjust is None else y_adjust),
        xanchor=xanchor,
        yanchor=yanchor,
        text=text,
    )


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


class LeafPlot(Plot):
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

    plots: List[Plot] = field(init=False)

    def __post_init__(self) -> None:
        self.plots = self.make_plots()

    # want #abstractmethod but MyPy doesn't support abstract data classes
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


class RootPlot:
    """Top-level plot."""

    @property
    @abstractmethod
    def plots(self) -> List[Plot]:
        pass

    @abstractmethod
    def layout(self) -> go.Figure:
        pass

    @abstractmethod
    def add_titles(self, fig: go.Figure) -> None:
        pass

    def render(self) -> None:
        """Create fig and render subplots."""
        fig: go.Figure = self.layout()

        for plot in self.plots:
            plot.layout_axes(fig)

        self.add_titles(fig)

        for plot in self.plots:
            plot.render(fig)

        fig.show(config=dict(displayModeBar=False, showAxisDragHandles=False))
