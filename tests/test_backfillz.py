"""Test module for backfillz."""

import pytest
from tests.generate_sample_fit import generate_fit, Stan

from backfillz.core import Backfillz
from backfillz.slice_histogram import SliceHistogram
from backfillz.theme import demo_1
from backfillz.trace_dial import TraceDial


@pytest.fixture(scope='session')
def stan() -> Stan:
    """Stan model shared by all tests."""
    return generate_fit()


def test_sample_fit(stan: Stan) -> None:
    """Backfillz object can be created."""
    Backfillz(stan.fit)
    file = "expected_backfillz"
#    stan.save(file)
    expected_stan = Stan.load(file)
    print(str(expected_stan))
    print(str(stan))
    assert expected_stan.equal(stan)


def test_plot_slice_histogram(stan: Stan) -> None:
    """Slice histogram plot is generated without error."""
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1, False)
    SliceHistogram.plot(backfillz)


def test_trace_dial(stan: Stan) -> None:
    """Trace dial plot is generated without error."""
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1, False)
    TraceDial.plot(backfillz)
