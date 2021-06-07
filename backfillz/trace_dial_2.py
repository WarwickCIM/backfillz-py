from dataclasses import dataclass
from math import ceil, floor
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.core import Backfillz, ParameterSlices, Props, Slice
from backfillz.plot import Plot, RootPlot, segment, VerticalSubplots
from backfillz.slice_histograms import SliceHistogram


@dataclass
class TraceDialHistogram(SliceHistogram):
    """Slice histogram for trace dial plot."""

    @property
    def xaxis_props(self) -> Props:
        if self.n_slc == len(self.data.slcs) - 1:
            return dict(side='top')
        else:
            return dict(visible=False)


@dataclass
class SliceHistograms(VerticalSubplots):
    """Two slice histograms, one for burn in, one for rest of chain."""

    def make_plots(self) -> List[Plot]:
        return [
            TraceDialHistogram(
                axis_id=self.axis_ids[n],
                x_domain=self.x_domain,
                y_domain=segment(self.y_domain, len(self.data.slcs), n),
                data=self.data,
                theme=self.theme,
                slc=slc,
                n_slc=n,
                row=self.row + len(self.data.slcs) - 1 - n,
                col=self.col,
            )
            for n, slc in enumerate(self.data.slcs)
        ]


# Experiment to see if I can add axes directly, without involving make_subplots.
@dataclass
class TraceDial2(RootPlot):
    """Top-level trace dial plot for a given parameter."""

    data: ParameterSlices

    @property
    def title(self) -> str:
        return f"Pretzel plot for {self.data.param}"

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return self.polar_traces + [self.histogram]

    # one per chain
    @property
    def polar_traces(self) -> List[go.Scatterpolar]:
        return [
            go.Scatterpolar(
                theta=[n / self.data.n_iter * 270 for n in range(0, self.data.n_iter)],
                r=chain,
                line=dict(color=self.theme.palette[n]),
            )
            for n, chain in enumerate(self.data.chains)
        ]

    @property
    def histogram(self) -> go.Histogram:
        return go.Histogram(
            x=[x for xs in self.data.chain_slices(self.data.slcs[0]) for x in xs],
            xbins=dict(start=floor(self.data.min_sample), end=ceil(self.data.max_sample), size=1),
            marker=dict(
                color=self.theme.bg_colour,
                line=dict(color=self.theme.fg_colour, width=1)
            ),
            histnorm='probability',
            xaxis='x2',
            yaxis='y2',
        )

    # Override render to customise layout.
    def render(self) -> None:
        """Create fig and render subplots."""
        fig: go.Figure = go.Figure(
            layout=go.Layout(
                title=self.title,
                titlefont=dict(size=30),
                plot_bgcolor=self.theme.bg_colour,
                showlegend=False,
                xaxis=dict(
                    domain=[0, 1],
                    anchor='y',
                ),
                yaxis=dict(
                    domain=[0, 1],
                    anchor='x',
                ),
                xaxis2=dict(
                    domain=[0.5, 1],
                    anchor='y2',
                ),
                yaxis2=dict(
                    domain=[0.75, 1],
                    anchor='x2',
                ),
                xaxis3=dict(
                    domain=[0.5, 1],
                    anchor='y3',
                ),
                yaxis3=dict(
                    domain=[0.5, 0.75],
                    anchor='x3',
                ),
                polar=dict(
                    sector=[0, 270],
                    hole=0.3,
                    bgcolor=self.theme.bg_colour,
                )
            )
        )
        print(fig.layout)

        for el in self.plot_elements:
            fig.add_trace(el)

        fig.show(config=dict(displayModeBar=False, showAxisDragHandles=False))

    @staticmethod
    def plot(backfillz: Backfillz, save_plot: bool = False) -> None:
        slcs: List[Slice] = [Slice(0.0, 0.04), Slice(0.4, 1)]  # how to decide
        param: str = backfillz.params[0]  # pick first parameter for now (mu)
        data: ParameterSlices = ParameterSlices(
            slcs=slcs,
            param=param,
            chains=backfillz.iter_chains(param),
            max_sample=np.amax(backfillz.mcmc_samples[param]),
            min_sample=np.amin(backfillz.mcmc_samples[param]),
        )

        TraceDial2(backfillz.theme, data).render()
