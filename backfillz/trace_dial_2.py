from dataclasses import dataclass

import plotly.graph_objects as go

from backfillz.plot import RootPlot


# Experiment to see if I can add axes directly, without involving make_subplots.
@dataclass
class TraceDial(RootPlot):
    """Top-level trace dial plot for a given parameter."""

    # Override render to customise layout.
    def render(self) -> None:
        """Create fig and render subplots."""
        fig: go.Figure = go.Figure(
            layout=go.Layout(
                title=self.title,
                titlefont=dict(size=30),
                plot_bgcolor=self.theme.bg_colour,
                showlegend=False,
            )
        )

        for plot in self.plots:
            plot.layout_axes(fig)

        self.add_additional_titles(fig)

        for plot in self.plots:
            plot.render(fig)

        fig.show(config=dict(displayModeBar=False, showAxisDragHandles=False))
