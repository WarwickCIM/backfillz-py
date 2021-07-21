from dataclasses import dataclass
from typing import List, Sequence

from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import MCMCRun, ParameterData, segment
from backfillz.plot import fresh_axis_id, LeafPlot, Plot, RootPlot
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

    @property
    def plot_elements(self) -> go.Scatter:
        pass


@dataclass
class SpiralRow(LeafPlot[ParameterSteps]):
    """Row of spiral plots for chain, one for each step."""

    n_chain: int

    @property
    def plot_elements(self) -> Sequence[BaseTraceType]:
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
                axis_id=fresh_axis_id(),
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
