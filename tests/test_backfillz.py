"""Test module for backfillz."""

import pytest

from backfillz import Backfillz
from backfillz.example.eight_schools import generate_fit
from backfillz.stan import Stan
from backfillz.theme import demo_1


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
    backfillz = Backfillz(stan.fit, verbose=True)
    backfillz.set_theme(demo_1)
    backfillz.plot_slice_histogram('mu')


# @pytest.mark.skip(reason="temporarily disable")
def test_trace_dial(stan: Stan) -> None:
    """Trace dial plot is generated without error."""
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1)
    backfillz.plot_trace_dial('mu')
