"""Test module for backfillz."""

import pytest
from tests.generate_sample_fit import generate_fit, Stan

from backfillz.core import Backfillz
from backfillz.theme import demo_1
from backfillz.trace_dial import TraceDial
from backfillz.trace_dial_2 import TraceDial2
from backfillz.trace_slice_histogram import TraceSliceHistogram


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


@pytest.mark.skip(reason="temporarily disable")
def test_trace_slice_histogram(stan: Stan) -> None:
    """Slice histogram plot is generated without error."""
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1, False)
    TraceSliceHistogram.plot(backfillz)


# @pytest.mark.skip(reason="temporarily disable")
def test_trace_dial(stan: Stan) -> None:
    """Trace dial plot is generated without error."""
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1, False)
    TraceDial.plot(backfillz)
