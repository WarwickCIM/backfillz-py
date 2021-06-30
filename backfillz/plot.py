from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from plotly.basedatatypes import BaseTraceType  # type: ignore
from plotly.colors import unlabel_rgb  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from backfillz.data import ParameterSlices, Props
from backfillz.theme import BackfillzTheme


class AbstractMethodError(NotImplementedError):
    """MyPy doesn't support abstract data classes yet (https://github.com/python/mypy/issues/5374)."""

    pass


# strings assigned as axis id suffixes by Plotly; empty for first subplot
AxisId = str
# Plotly subplot specs; 2D array of dictionaries
Specs = List[List[object]]


def default_config() -> Props:
    """Preferred settings for Plotly figure."""
    return dict(displayModeBar=False, showAxisDragHandles=False)


def cols(xss: Specs) -> int:
    """Length of the inner lists."""
    ns: Set[int] = set(map(len, xss))
    assert len(ns) == 1
    return min(ns)


def scale(factor: float, xs: List[float]) -> List[float]:
    """Element-wise product."""
    return [x * factor for x in xs]


def segment(domain: Tuple[float, float], n: int, m: int) -> Tuple[float, float]:
    """Break supplied "domain" into n equal-sized segments, and return the mth."""
    start, end = domain
    width = (end - start) / n
    return start + m * width, start + (m + 1) * width


def alpha(colour: str, a: float) -> str:
    """Add an alpha component to a colour represented as an RGB string."""
    rgb: tuple[int, int, int] = unlabel_rgb(colour)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]},{a})"


def annotate(
    fig: go.Figure,
    font_size: int,
    at: Tuple[float, float],
    xanchor: Literal['left', 'right'],
    yanchor: Literal['top', 'bottom'],
    y_adjust: Optional[float],  # additional normalised offet of text relative to plot
    text: str,
    textangle: int = 0,
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
        textangle=textangle
    )


@dataclass
class Plot:
    """Base class providing common subplot functionality."""

    x_domain: Tuple[float, float]  # left/right edges normalised to [0, 1]
    y_domain: Tuple[float, float]  # top/bottom edges normalised to [0, 1]
    row: int
    col: int
    data: ParameterSlices
    theme: BackfillzTheme

    def layout_axes(self, fig: go.Figure) -> None:
        raise AbstractMethodError()

    def render(self, fig: go.Figure) -> None:
        """Render me into fig."""
        raise AbstractMethodError()

    @property
    def top_left(self) -> Tuple[float, float]:
        return self.x_domain[0], self.y_domain[1]


@dataclass
class LeafPlot(Plot):
    """A leaf subplot."""

    # Axis ids (and annotation ids) need hand-configuration to match assignment by Plotly.
    axis_id: AxisId

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        raise AbstractMethodError()

    def render(self, fig: go.Figure) -> None:
        for el in self.plot_elements:
            fig.add_trace(el, self.row, self.col)

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
        return 'xaxis' + self.axis_id

    @property
    def yaxis_id(self) -> str:
        """My Plotly-assigned y-axis id."""
        return 'yaxis' + self.axis_id

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
class LeafPlotNoAxes(Plot):
    """A leaf subplot of "domain" type, i.e. with no axes."""

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        raise AbstractMethodError()

    def render(self, fig: go.Figure) -> None:
        for el in self.plot_elements:
            fig.add_trace(el)

    def layout_axes(self, fig: go.Figure) -> None:
        pass


@dataclass
class VerticalSubplots(Plot):
    """A collection of vertically arranged subplots."""

    axis_ids: List[AxisId]  # one per subplots
    plots: List[Plot] = field(init=False)

    def __post_init__(self) -> None:
        self.plots = self.make_plots()

    def make_plots(self) -> List[Plot]:
        """My subplots."""
        raise AbstractMethodError()

    def layout_axes(self, fig: go.Figure) -> None:
        """Ask each subplot to configure its axes."""
        for plot in self.plots:
            plot.layout_axes(fig)

    def render(self, fig: go.Figure) -> None:
        """Render subplots into fig."""
        for n, plot in enumerate(self.plots):
            plot.render(fig)


# Should consolidate some of the commonality with Plot.
@dataclass
class RootPlot:
    """Top-level plot container."""

    theme: BackfillzTheme
    verbose: bool

    @property
    def plots(self) -> List[Plot]:
        raise AbstractMethodError()

    def grid_specs(self, fig: go.Figure) -> Specs:
        raise AbstractMethodError()

    @property
    def title(self) -> str:
        """Title for overall figure."""
        raise AbstractMethodError()

    def add_additional_titles(self, fig: go.Figure) -> None:
        annotate(
            fig, 14, (1, -0.03), 'right', 'top', None,  # leave room for an x-axis, if needed
            "Backfillz-py by CIM, University of Warwick and The Alan Turing Institute"
        )

    def render(self) -> go.Figure:
        """Create fig and render subplots."""
        fig: go.Figure = go.Figure(
            layout=go.Layout(
                title=self.title,
                titlefont=dict(size=30),
                plot_bgcolor=self.theme.bg_colour,
                showlegend=False,
            )
        )

        specs: Specs = self.grid_specs(fig)

        make_subplots(
            rows=len(specs),
            cols=cols(specs),
            figure=fig,
            specs=specs,
            horizontal_spacing=0,
            vertical_spacing=0,
            print_grid=self.verbose,
        )

        for plot in self.plots:
            plot.layout_axes(fig)

        self.add_additional_titles(fig)

        for plot in self.plots:
            plot.render(fig)

        return fig
