"""Test module for backfillz."""

from typing import List

import plotly.graph_objects as go
import pytest

from backfillz import plot_slice_histogram
from backfillz.core import Backfillz
from backfillz.example.eight_schools import generate_fit
from backfillz.plot import show
from backfillz.stan import Stan
from backfillz.theme import demo_1
from backfillz.trace_dial import TraceDial
from backfillz.trace_slice_histogram import TraceSliceHistogram


@pytest.fixture(scope='session')
def stan() -> Stan:
    """Stan model shared by all tests."""
    return generate_fit()


# @pytest.mark.skip(reason="temporarily disable")
def test_sample_fit(stan: Stan) -> None:
    """Backfillz object can be created."""
    Backfillz(stan.fit)
    file = "expected_sample_fit"
#    stan.save(file)
    expected_stan = Stan.load(file)
    print(str(expected_stan))
    print(str(stan))
    assert expected_stan.equal(stan)


# @pytest.mark.skip(reason="temporarily disable")
def test_trace_slice_histogram(stan: Stan) -> None:
    """Slice histogram plot is generated without error."""
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1, False)
    plot_slice_histogram(backfillz, 'mu')


# @pytest.mark.skip(reason="temporarily disable")
def test_trace_dial(stan: Stan) -> None:
    """Trace dial plot is generated without error."""
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1, False)
    figs: List[go.Figure] = TraceDial.figs(backfillz)
    for fig in figs:
        show(fig)
