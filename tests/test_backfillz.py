"""Test module for backfillz."""

from tests.generate_sample_fit import generate_fit, Stan

from backfillz.core import Backfillz
from backfillz.slice_histogram import SliceHistogram
from backfillz.theme import demo_1
from backfillz.trace_dial import TraceDial


class TestBackfillz:
    stan: Stan


def test_sample_fit() -> None:
    """Backfillz object can be created."""
    stan = generate_fit()
    Backfillz(stan.fit)
    file = "expected_backfillz"
#    stan.save(file)
    expected_stan = Stan.load(file)
    print(str(expected_stan))
    print(str(stan))
    assert expected_stan.equal(stan)


def test_plot_slice_histogram() -> None:
    """Slice histogram plot is generated without error."""
    stan = generate_fit()
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1, False)
    SliceHistogram.plot(backfillz)


def test_trace_dial() -> None:
    """Trace dial plot is generated without error."""
    stan = generate_fit()
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1, False)
    TraceDial.plot(backfillz)


if __name__ == '__main__':
    test_sample_fit()
    test_plot_slice_histogram()
    test_trace_dial()
