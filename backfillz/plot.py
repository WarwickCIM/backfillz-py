from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from plotly.basedatatypes import BaseTraceType  # type: ignore
from plotly.colors import unlabel_rgb  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import Domain, ParameterSlices, Point, Props
from backfillz.theme import BackfillzTheme


class AbstractMethodError(NotImplementedError):
    """MyPy doesn't support abstract data classes yet (https://github.com/python/mypy/issues/5374)."""

    pass


AxisId = str
axis_count = 2  # start with 2 for consistently with Plotly


def fresh_axis_id() -> str:
    """Allocate an axis id that hasn't been used elsewhere."""
    global axis_count
    axis_id = axis_count
    axis_count = axis_count + 1
    return str(axis_id)


def default_config() -> Props:
    """Preferred settings for Plotly figure."""
    return dict(displayModeBar=False, showAxisDragHandles=False)


def scale(factor: float, xs: List[float]) -> List[float]:
    """Element-wise product."""
    return [x * factor for x in xs]


def segment(domain: Domain, n: int, m: int) -> Domain:
    """Break supplied "domain" into n equal-sized segments, and return the mth."""
    start, end = domain
    width = (end - start) / n
    return start + m * width, start + (m + 1) * width


def alpha(colour: str, a: float) -> str:
    """Add an alpha component to a colour represented as an hex string without an alpha component."""
    rgb: tuple[int, int, int] = unlabel_rgb(colour)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]},{a})"


def annotate(
    fig: go.Figure,
    font_size: int,
    at: Point,
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

    x_domain: Domain  # left/right edges normalised to [0, 1]
    y_domain: Domain  # top/bottom edges normalised to [0, 1]
    data: ParameterSlices
    theme: BackfillzTheme

    def layout_axes(self, fig: go.Figure) -> None:
        raise AbstractMethodError()

    def render(self, fig: go.Figure) -> None:
        """Render me into fig."""
        raise AbstractMethodError()

    @property
    def top_left(self) -> Point:
        return self.x_domain[0], self.y_domain[1]


@dataclass
class LeafPlot(Plot):
    """A leaf subplot."""

    # Either generated using fresh_axis_id, or '' to mean the figure's default axes.
    axis_id: AxisId

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        raise AbstractMethodError()

    def render(self, fig: go.Figure) -> None:
        for el in self.plot_elements:
            fig.add_trace(el)

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
        fig.update_layout({
            self.xaxis_id: dict(anchor='y' + self.axis_id),
            self.yaxis_id: dict(anchor='x' + self.axis_id)
        })
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
class AggregatePlot(Plot):
    """A collection of subplots."""

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
class RootPlot(AggregatePlot):
    """Top-level plot container."""

    verbose: bool

    @property
    def title(self) -> str:
        """Title for overall figure."""
        raise AbstractMethodError()

    def add_additional_titles(self, fig: go.Figure) -> None:
        annotate(
            fig, 14, (1, -0.03), 'right', 'top', None,  # leave room for an x-axis, if needed
            "Backfillz-py by CIM, University of Warwick and The Alan Turing Institute"
        )

    @property
    def layout_props(self) -> Props:
        return dict()

    def make_fig(self) -> go.Figure:
        """Create fig and render subplots."""
        fig: go.Figure = go.Figure(
            layout=go.Layout(
                title=self.title,
                titlefont=dict(size=30),
                plot_bgcolor=self.theme.bg_colour,
                showlegend=False,
                **self.layout_props,
            )
        )

        self.layout_axes(fig)
        self.add_additional_titles(fig)
        self.render(fig)
        return fig
