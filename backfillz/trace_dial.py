from dataclasses import dataclass
from backfillz.core import Backfillz, ParameterSlices
from backfillz.plot import LeafPlot, RootPlot, VerticalSubplots
from backfillz.theme import BackfillzTheme

import plotly.graph_objects as go  # type: ignore


@dataclass
class DialPlot(LeafPlot):
    pass


@dataclass
class Histograms(VerticalSubplots):
    pass


@dataclass
class TraceDial(RootPlot):
    """Top-level plot, for a given parameter and chain."""

    data: ParameterSlices
    theme: BackfillzTheme

    @property
    def plots(self):
        return [self.dial_plot, self.histograms]

    @property
    def dial_plot(self) -> DialPlot:
        return DialPlot(
            axis_ids=[None],
            # top-right quadrant:
            x_domain=(0.5, 1.0),
            y_domain=(0.5, 1.0),
            row=1,
            col=2,
            data=self.data,
            theme=self.theme,
        )

    @property
    def histograms(self) -> Histograms:
        return Histograms(
            axis_ids=[2, 3],
            x_domain=(0, 1.0),
            y_domain=(0, 1.0),
            row=1,
            col=1,
            data=self.data,
            theme=self.theme,
        )

    def layout(self) -> go.Figure:
        pass

    def add_title(self, fig: go.Figure) -> None:
        pass

    @staticmethod
    def plot(backfillz: Backfillz) -> None:
        pass
