from dataclasses import dataclass
from typing import List

from plotly.basedatatypes import BaseTraceType  # type: ignore
import plotly.graph_objects as go  # type: ignore

from backfillz.data import MCMCRun, ParameterSlices, Props, scale, segment, Slice
from backfillz.plot import AggregatePlot, annotate, fresh_axis_id, LeafPlot, Plot, RootPlot
from backfillz.slice_histograms import SliceHistogram
from backfillz.theme import BackfillzTheme


@dataclass
class TracePlot(LeafPlot[ParameterSlices]):
    """Left-hand component."""

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return self.traces + self.boxes

    # one per chain
    @property
    def traces(self) -> List[go.Scatter]:
        return [
            go.Scatter(x=chain, y=[*range(0, self.data.n_iter)], line=dict(color=self.theme.palette[n]))
            for n, chain in enumerate(self.data.chains)
        ]

    # one per slice
    @property
    def boxes(self) -> List[go.Scatter]:
        return [
            go.Scatter(
                x=[self.data.min_sample] * 2 + [self.data.max_sample] * 2 + [self.data.min_sample],
                y=scale(self.data.n_iter, [slc.lower, slc.upper, slc.upper, slc.lower, slc.lower]),
                mode='lines',
                line=dict(width=2, color=self.theme.fg_colour),
            )
            for slc in self.data.slcs
        ]

    @property
    def xaxis_props(self) -> Props:
        return dict(range=[self.data.min_sample, self.data.max_sample])

    @property
    def yaxis_props(self) -> Props:
        return dict(range=[0, self.data.n_iter])


@dataclass
class JoiningSegments(LeafPlot[ParameterSlices]):
    """Middle component."""

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        return self.segments + [self.y_labels]

    # one per slice
    @property
    def segments(self) -> List[go.Scatter]:
        return [
            go.Scatter(
                xaxis='x' + self.axis_id,
                x=[0, 1, 1, 0],
                y=scale(self.data.n_iter, [slc.lower, lower, upper, slc.upper]),
                mode='lines',
                line=dict(color=self.theme.fg_colour, width=1),
                fill='toself',
                fillcolor='rgba(240,240,240,255)',
            )
            for n, slc in enumerate(self.data.slcs, start=1)
            for lower, upper in [((n - 1) / len(self.data.slcs), n / len(self.data.slcs))]
        ]

    # one numerical marker per slice delimiter
    @property
    def y_labels(self) -> go.Scatter:
        y = self.slice_delimiters
        return go.Scatter(
            xaxis='x' + self.axis_id,
            x=[0] * len(y),
            y=y,
            mode='text',
            text=[int(y) for y in y],
            textposition='middle right',
        )

    @property
    def slice_delimiters(self) -> List[float]:
        """Unique slice start/end points, expressed in iterations."""
        delims: List[float] = [*{*[y for slc in self.data.slcs for y in [slc.lower, slc.upper]]}]
        return scale(self.data.n_iter, delims)

    @property
    def xaxis_props(self) -> Props:
        return dict(rangemode='nonnegative', visible=False)

    @property
    def yaxis_props(self) -> Props:
        return dict(
            range=[0, self.data.n_iter],
            tickmode='array',
            tickvals=self.slice_delimiters,
            showticklabels=False
        )


@dataclass
class SliceHistograms(AggregatePlot[ParameterSlices]):
    """One slice histogram per slice."""

    def make_plots(self) -> List[Plot[ParameterSlices]]:
        return [
            SliceHistogram(
                axis_id=fresh_axis_id(),
                x_domain=self.x_domain,
                y_domain=segment(self.y_domain, len(self.data.slcs), n),
                data=self.data,
                theme=self.theme,
                slc=slc,
                n_slc=n,
            )
            for n, slc in enumerate(self.data.slcs)
        ]


@dataclass
class TraceSliceHistogram(RootPlot[ParameterSlices]):
    """Trace slice histogram plot for a given parameter."""

    left_w = 0.4  # width of trace plot
    middle_w = 0.2  # width of joining segments

    def make_plots(self) -> List[Plot[ParameterSlices]]:
        return [self.trace_plot, self.joining_segments, self.density_plots]

    @property
    def trace_plot(self) -> TracePlot:
        return TracePlot(
            axis_id='',  # Plotly default axes
            x_domain=(0, self.left_w),
            y_domain=(0, 1.0),
            data=self.data,
            theme=self.theme,
        )

    @property
    def joining_segments(self) -> JoiningSegments:
        return JoiningSegments(
            axis_id=fresh_axis_id(),
            x_domain=(self.left_w, self.left_w + self.middle_w),
            y_domain=(0, 1.0),
            data=self.data,
            theme=self.theme,
        )

    @property
    def density_plots(self) -> SliceHistograms:
        return SliceHistograms(
            x_domain=(self.left_w + self.middle_w, 1),
            y_domain=(0, 1.0),
            data=self.data,
            theme=self.theme,
        )

    @property
    def title(self) -> str:
        return f"Trace slice histogram of {self.data.param}"

    def add_additional_titles(self, fig: go.Figure) -> None:
        super().add_additional_titles(fig)
        annotate(fig, 16, self.trace_plot.top_left, 'left', 'bottom', None, "Trace Plot With Slices")
        # oof -- adjust for x-axis
        annotate(fig, 16, self.density_plots.top_left, 'left', 'bottom', 0.03, "Density Plots for Slices")

    @staticmethod
    def fig(mcmc_run: MCMCRun, theme: BackfillzTheme, verbose: bool, param: str) -> go.Figure:
        """Create a slice histogram."""
        slcs: List[Slice] = [Slice(0.028, 0.04), Slice(0.1, 0.2), Slice(0.4, 0.9)]
        return TraceSliceHistogram(
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            data=ParameterSlices(mcmc_run, param, slcs),
            theme=theme,
            verbose=verbose,
        ).make_fig()
