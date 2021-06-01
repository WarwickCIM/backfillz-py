from dataclasses import dataclass
from backfillz.core import Backfillz, ParameterSlices
from backfillz.plot import LeafPlot, RootPlot, VerticalSubplots
from backfillz.theme import BackfillzTheme


@dataclass
class DialPlot(LeafPlot):
    pass


@dataclass
class Histograms(VerticalSubplots):
    pass


@dataclass
class TraceDial(RootPlot):
    """Top-level plot, for a given parameter and chain."""

    theme: BackfillzTheme
    data: ParameterSlices

    @property
    def plots(self):
        return [self.dial_plot, self.histograms]

    @property
    def dial_plot(self) -> DialPlot:
        return DialPlot(
            axis_ids=[None],
            x_domain=(0, 1.0),
            y_domain=(0, 1.0),
            row=1,
            col=1,
            data=self.data,
            theme=self.theme,
        )

    @property
    def histograms(self) -> Histograms:
        return Histograms()

    @staticmethod
    def plot(backfillz: Backfillz) -> None:
        pass
