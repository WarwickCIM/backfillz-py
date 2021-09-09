from dataclasses import dataclass
from math import floor, pi
from typing import List, Sequence, Tuple

import plotly.graph_objects as go  # type: ignore

from backfillz.data import Axis, Domain, MCMCRun, ParameterData, Props, segment
from backfillz.plot import AggregatePlot, annotate, fresh_axis_id, LeafPlot, RootPlot, spiral_plot
from backfillz.theme import alpha, BackfillzTheme


@dataclass
class ParameterSteps(ParameterData):
    """Data visualised by spiral stream plot."""

    steps: List[int]


@dataclass
class SpiralPlot(LeafPlot[ParameterSteps]):
    """Spiral plot for chain and step."""

    n_chain: int
    step: int

    @property
    def plot_elements(self) -> List[go.Scatter]:
        return [self.spiral_plot]

    @property
    def angular_axis(self) -> Axis:
        return Axis((0, self.data.n_iter), (0.5 * pi, 2 * pi * 3))

    @property
    def spiral_plot(self) -> go.Scatter:
        chain: List[float] = self.data.variance(self.n_chain, self.step)
        xs: List[int] = [*range(0, self.data.n_iter)]
        y_range: Domain = min(chain), max(chain)
        # plot variance (which is always positive) and its negation in the same plot, and close into a polygon
        xs1, ys1 = spiral_plot(xs, chain, self.angular_axis, Axis(y_range, (0.5, 1)), 1 / (2 * pi))
        xs2, ys2 = spiral_plot(xs, chain, self.angular_axis, Axis(y_range, (0.5, 0)), 1 / (2 * pi))
        return go.Scatter(
            x=xs1 + xs2[::-1],
            y=ys1 + ys2[::-1],
            fill='toself',
            fillcolor=alpha(self.theme.palette[self.n_chain], 0.5),
            line=dict(width=0.5, color=self.theme.palette[self.n_chain]),
            xaxis='x' + self.axis_id,
            yaxis='y' + self.axis_id,
        )

    @property
    def overall_range(self) -> Tuple[int, int]:
        """Number of rotations determines x and y range (expressed in unit radii)."""
        _, end = self.angular_axis.domain
        rotations: int = floor(end / (2 * pi) + 1)
        return (-rotations, rotations)

    @property
    def xaxis_props(self) -> Props:
        return dict(visible=False, range=self.overall_range)

    @property
    def yaxis_props(self) -> Props:
        return dict(visible=False, range=self.overall_range)


@dataclass
class SpiralRow(AggregatePlot[ParameterSteps]):
    """Row of spiral plots for chain, one for each step."""

    n_chain: int

    def make_plots(self) -> Sequence[SpiralPlot]:
        return [
            SpiralPlot(
                data=self.data,
                theme=self.theme,
                axis_id=fresh_axis_id(),
                x_domain=segment(self.x_domain, len(self.data.steps), n),
                y_domain=self.y_domain,
                n_chain=self.n_chain,
                step=step,
            )
            for n, step in enumerate(self.data.steps)
        ]


@dataclass
class SpiralStream(RootPlot[ParameterSteps]):
    """Spiral stream plot for a given parameter; one spiral row per chain."""

    def make_plots(self) -> Sequence[SpiralRow]:
        return [
            SpiralRow(
                x_domain=(0, 1),
                y_domain=segment(self.y_domain, len(self.data.chains), n),
                data=self.data,
                theme=self.theme,
                n_chain=n,
            )
            for n, _ in enumerate(self.data.chains)
        ]

    @property
    def title(self) -> str:
        return f"Spiral stream plot for {self.data.param}"

    def add_additional_titles(self, fig: go.Figure) -> None:
        super().add_additional_titles(fig)
        annotate(fig, 14, (0.5, 0), 'center', 'middle', None, "Variance")
        annotate(fig, 14, (0, 0.5), 'right', 'middle', None, "Chain", textangle=-90)

    @property
    def layout_props(self) -> Props:
        # ensure each individual spiral plot is square; see trace_dial
        length: int = 800
        return dict(width=length, height=length * len(self.data.chains) / len(self.data.steps))

    @staticmethod
    def fig(
        mcmc_run: MCMCRun,
        theme: BackfillzTheme,
        verbose: bool,
        param: str,
        steps: List[int]
    ) -> go.Figure:
        """Create a spiral stream plot."""
        return SpiralStream(
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            data=ParameterSteps(mcmc_run, param, steps),
            theme=theme,
            verbose=verbose,
        ).make_fig()
