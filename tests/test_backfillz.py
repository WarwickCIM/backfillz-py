"""Test module for backfillz."""

from tests.generate_sample_fit import generate_fit, Stan

from backfillz.Backfillz import Backfillz
from backfillz.BackfillzTheme import demo_1
from backfillz.PlotSliceHistogram import plot_slice_histogram


def test_sample_fit() -> None:
    """Backfillz object can be created."""
    stan = generate_fit()
    file = "expected_backfillz"
#    stan.save(file)
    Backfillz(stan.fit)
    expected_stan = Stan.load(file)
    print(str(expected_stan))
    print(str(stan))
    assert expected_stan.equal(stan)


def test_plot_slice_histogram() -> None:
    """Slice histogram plot is correctly generated."""
    stan = generate_fit()
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_1, False)
    plot_slice_histogram(backfillz)


if __name__ == '__main__':
    test_sample_fit()
    test_plot_slice_histogram()
