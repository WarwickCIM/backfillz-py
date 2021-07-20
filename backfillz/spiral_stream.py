from dataclasses import dataclass
from math import nan
from typing import List

import numpy as np
from plotly.basedatatypes import BaseTraceType  # type: ignore

from backfillz.plot import AggregatePlot, fresh_axis_id, LeafPlot, Plot, RootPlot, segment


@dataclass
class SpiralRow(LeafPlot):
    """Row of spiral plots for chain with index n."""

    n: int

    @property
    def plot_elements(self) -> List[BaseTraceType]:
        steps = [3, 8, 15]
        spiral_points = [nan] * self.data.n_iter
        for step in steps:
            for i in self.data.n_iter:
                if i >= step:
                    klw = i - step
                else:
                    klw = 1

                if i <= (self.data.n_iter - step):
                    khg = i + step
                else:
                    khg = self.data.n_iter

            spiral_points[i] = np.var(self.data.chains[self.n][klw:khg])


class SpiralRows(AggregatePlot):
    """One spiral row per chain."""

    def make_plots(self) -> List[Plot]:
        return [
            SpiralRow(
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
class SpiralStream(RootPlot):
    """Spiral stream plot for a given parameter."""

    def make_plots(self) -> List[Plot]:
        return [self.spiral_rows]

    @property
    def spiral_rows(self) -> List[SpiralRows]:
        return [
            SpiralRows(
                axis_id=fresh_axis_id(),
                x_domain=(0, 1),
                y_domain=segment(self.y_domain, len(self.data.chains), n),
                data=self.data,
                theme=self.theme,
            )
            for n, _ in enumerate(self.data.chains)
        ]
