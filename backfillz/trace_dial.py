from backfillz.plot import LeafPlot, RootPlot, VerticalSubplots


class DialPlot(LeafPlot):
    pass


class Histograms(VerticalSubplots):
    pass


class TraceDial(RootPlot):
    """Top-level plot, for a given parameter and chain."""

    @property
    def plots(self):
        return [self.dial_plot, self.histograms]

    @property
    def dial_plot(self) -> DialPlot:
        return DialPlot()

    @property
    def histograms(self) -> Histograms:
        return Histograms()
