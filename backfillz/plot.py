from dataclasses import dataclass, field
from math import cos, floor, log10, pi, sin
from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, Tuple, TypeVar

from plotly.basedatatypes import BaseTraceType  # type: ignore
from plotly.colors import hex_to_rgb  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import Domain, ParameterData, Point, Props
from backfillz.theme import BackfillzTheme


class AbstractMethodError(NotImplementedError):
    """MyPy doesn't support abstract data classes yet (https://github.com/python/mypy/issues/5374)."""

    pass


AxisId = str
axis_count = 2  # start with 2 for consistently with Plotly

T = TypeVar('T', bound='ParameterData')


@dataclass
class Plot(Generic[T]):
    """Base class providing common subplot functionality."""

    x_domain: Domain  # left/right edges normalised to [0, 1]
    y_domain: Domain  # top/bottom edges normalised to [0, 1]
    data: T
    theme: BackfillzTheme

    def layout_axes(self, fig: go.Figure) -> None:
        raise AbstractMethodError()

    def render(self, fig: go.Figure) -> None:
        """Render me into fig."""
        raise AbstractMethodError()

    @property
    def top_left(self) -> Point:
        return self.x_domain[0], self.y_domain[1]

    @property
    def bottom_right(self) -> Point:
        return self.x_domain[1], self.y_domain[0]


@dataclass
class LeafPlot(Plot[T]):
    """A leaf subplot."""

    # Either generated using fresh_axis_id, or '' to mean the figure's default axes.
    axis_id: AxisId

    @property
    def plot_elements(self) -> Sequence[BaseTraceType]:
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
        """My Plotly x-axis id."""
        return 'xaxis' + self.axis_id

    @property
    def yaxis_id(self) -> str:
        """My Plotly y-axis id."""
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
class AggregatePlot(Plot[T]):
    """A collection of subplots."""

    plots: Sequence[Plot[T]] = field(init=False)

    def __post_init__(self) -> None:
        self.plots = self.make_plots()

    def make_plots(self) -> Sequence[Plot[T]]:
        """My subplots."""
        raise AbstractMethodError()

    def layout_axes(self, fig: go.Figure) -> None:
        """Ask each subplot to configure its axes."""
        for plot in self.plots:
            plot.layout_axes(fig)

    def render(self, fig: go.Figure) -> None:
        """Render subplots into fig."""
        for plot in self.plots:
            plot.render(fig)


@dataclass
class RootPlot(AggregatePlot[T]):
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
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                bargap=0,
                **self.layout_props,
            )
        )

        self.layout_axes(fig)
        self.add_additional_titles(fig)
        self.render(fig)
        return fig


@dataclass
class Axis:
    """Map a range into a domain."""

    range: Domain
    domain: Domain

    # Don't require that r_start <= x <= r_end.
    def translate(self, xs: Sequence[float]) -> Sequence[float]:
        r_start, r_end = self.range
        d_start, d_end = self.domain
        return [(x - r_start) / (r_end - r_start) * (d_end - d_start) + d_start for x in xs]


def normalise(xs: Sequence[float], domain: Domain) -> Axis:
    """Map a data range into a domain."""
    return Axis((min(xs), max(xs)), domain)


def fresh_axis_id() -> str:
    """Allocate an axis id that hasn't been used elsewhere."""
    global axis_count
    axis_id = axis_count
    axis_count = axis_count + 1
    return str(axis_id)


def default_config() -> Props:
    """Preferred settings for Plotly figure."""
    return dict(displayModeBar=False, showAxisDragHandles=False)


def alpha(hex_colour: str, a: float) -> str:
    """Add an alpha component to a colour represented as a hex string without an alpha component."""
    rgb: tuple[int, int, int] = hex_to_rgb(hex_colour)
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"


def annotate(
    fig: go.Figure,
    font_size: int,
    at: Point,
    xanchor: Literal['left', 'center', 'right'],
    yanchor: Literal['top', 'middle', 'bottom'],
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


def left_vertical_title(fig: go.Figure, plot: Plot[T], title: str) -> None:
    """Add vertical title to left of plot."""
    annotate(fig, 14, plot.top_left, 'right', 'top', None, title, textangle=-90)


def background_rect(plot: Plot[T], fillcolor: str) -> Props:
    """Shaded background for a plot, as a Plotly shape that can be added to layout.shapes."""
    x0, y0 = plot.top_left
    x1, y1 = plot.bottom_right
    return dict(
        type='rect',
        xref='paper', yref='paper',
        x0=x0, y0=y0,
        x1=x1, y1=y1,
        fillcolor=fillcolor,
        layer='below',
        line_width=0,
    )


def tick_every(ticks_per_circle: int, angular_axis: Axis) -> int:
    """Tick gap in range units, based on desired approximate number of ticks per circle."""
    dom_start, dom_end = angular_axis.domain
    num_ticks: float = (dom_end - dom_start) / (2 * pi) * ticks_per_circle
    start, end = angular_axis.range
    tick_gap: float = (end - start) / num_ticks
    return int(round(tick_gap, -int(floor(log10(abs(tick_gap))))))  # 1 sig fig


def spiral_plot(
    xs: Sequence[float],
    ys: Sequence[float],
    x_axis: Axis,
    y_axis: Axis,
    b: float,
) -> Tuple[List[float], List[float]]:
    """Plot along arithmetic spiral r = a + b * theta, via the supplied axes. 12 o'clock = 0.5 * pi."""
    assert len(xs) == len(ys)
    thetas = x_axis.translate(xs)
    ys_radial = [y + b * theta for theta, y in zip(thetas, y_axis.translate(ys))]
    rs_thetas: List[Tuple[float, float]] = [
        (cos(theta) * y, sin(theta) * y)
        for theta, y in zip(thetas, ys_radial)
    ]
    return [r for r, _ in rs_thetas], [theta for _, theta in rs_thetas]


def polar_plot(
    xs: Sequence[float],
    ys: Sequence[float],
    x_axis: Axis,
    y_axis: Axis
) -> Tuple[List[float], List[float]]:
    """A spiral plot with b = 0."""
    return spiral_plot(xs, ys, x_axis, y_axis, 0)
