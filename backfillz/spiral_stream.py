from dataclasses import dataclass
from math import floor, pi
from typing import List, Sequence

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import Domain, MCMCRun, ParameterData, Props, segment
from backfillz.plot import AggregatePlot, Axis, fresh_axis_id, LeafPlot, Plot, RootPlot, spiral_plot
from backfillz.theme import BackfillzTheme


@dataclass
class ParameterSteps(ParameterData):
    """Data being visualised by a spiral stream plot."""

    steps: List[int]


@dataclass
class SpiralPlot(LeafPlot[ParameterSteps]):
    """Spiral plot for chain and step."""

    n_chain: int
    step: int

    angular_domain: Domain = 0.5 * pi, 2 * pi * 3

    @property
    def angular_axis(self) -> Axis:
        return Axis((0, self.data.n_iter), SpiralPlot.angular_domain)

    @property
    def radial_axis(self) -> Axis:
        return Axis((self.data.min_sample, self.data.max_sample), (0, 1))

    @property
    def plot_elements(self) -> List[go.Scatter]:
        chain: np.ndarray = self.data.chains[self.n_chain]
        # xs, ys = spiral_plot(10, 2, (0, 8 * pi))

        theta_domain: Domain = (0, 2 * pi)
        theta_start, theta_end = theta_domain
        theta_incr: float = 2 * pi / 36  # 10-degree increments
        thetas: List[float] = [
            x * theta_incr
            for x in range(floor(theta_start / theta_incr), floor((theta_end - theta_start) / theta_incr) + 1)
        ]
        angular_axis: Axis = Axis((0, self.data.n_iter), theta_domain)
        thetas: List[float] = [angular_axis.translate(x) for x, _ in enumerate(chain)]
        ys_radial: List[float] = [self.radial_axis.translate(y) for y in chain]
        xs, ys = spiral_plot(ys_radial, 0, thetas)
        return [go.Scatter(
            x=xs,
            y=ys,
            line=dict(width=1, color=self.theme.palette[self.n_chain]),
            xaxis='x' + self.axis_id,
            yaxis='y' + self.axis_id,
        )]

    @property
    def xaxis_props(self) -> Props:
        return dict(visible=True)

    @property
    def yaxis_props(self) -> Props:
        return dict(visible=True)


@dataclass
class SpiralRow(AggregatePlot[ParameterSteps]):
    """Row of spiral plots for chain, one for each step."""

    n_chain: int

    def make_plots(self) -> Sequence[BaseTraceType]:
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

    def make_plots(self) -> Sequence[Plot[ParameterSteps]]:
        return self.spiral_rows

    @property
    def spiral_rows(self) -> Sequence[SpiralRow]:
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

    @property
    def layout_props(self) -> Props:
        # ensure each individual spiral plot is square; see trace_dial
        length: int = 600
        return dict(width=length, height=length * len(self.data.chains) / len(self.data.steps))

    @staticmethod
    def fig(mcmc_run: MCMCRun, theme: BackfillzTheme, verbose: bool, param: str) -> go.Figure:
        """Create a spiral stream plot."""
        steps: List[int] = [3, 8, 15]  # defaults for now
        return SpiralStream(
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            data=ParameterSteps(mcmc_run, param, steps),
            theme=theme,
            verbose=verbose,
        ).make_fig()
