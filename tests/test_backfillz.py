"""Test module for backfillz."""

import plotly.graph_objects as go  # type: ignore
import pytest

from backfillz import Backfillz
from backfillz.example.eight_schools import generate_fit
from backfillz.stan import Stan
from backfillz.theme import default, demo_1, demo_2


@pytest.fixture(scope='session')
def stan() -> Stan:
    """Stan model shared by all tests."""
    return generate_fit()


@pytest.fixture()
def compare_images(pytestconfig) -> bool:
    """Whether to compare generated images with stored expected images."""
    print("Printing something")
    return pytestconfig.getoption("compare_images") == "True"


# Plotly doesn't generate SVGs deterministically, so use PNGs instead.
def expect_fig(fig: go.Figure, filename: str, check: bool) -> None:
    """Check for pixel-for-pixel equivalence to stored image."""
    if check:
        found = fig.to_image(format="png")
        try:
            file = open(filename + ".png", "rb")
            expected = file.read()
            if expected != found:
                file_new = open(filename + ".new.png", "wb")
                file_new.write(found)
                assert False
            else:
                print(f"{filename}: image identical.")
        except FileNotFoundError:
            file_new = open(filename + ".png", "wb")
            file_new.write(found)
    else:
        print(f"{filename}: image not compared.")


# @pytest.mark.skip(reason="temporarily disable")
def test_sample_fit(stan: Stan) -> None:
    """Backfillz object can be created."""
    Backfillz(stan.fit)
    file = "tests/expected_sample_fit"
#    stan.save(file)
    expected_stan = Stan.load(file)
    print(str(expected_stan))
    print(str(stan))
    assert expected_stan.equal(stan)


# @pytest.mark.skip(reason="temporarily disable")
def test_trace_slice_histogram(stan: Stan, compare_images: bool) -> None:
    """Slice histogram plot is generated without error."""
    backfillz = Backfillz(stan.fit, verbose=True)
    backfillz.set_theme(demo_1)
    fig: go.Figure = backfillz.plot_slice_histogram('mu')
    expect_fig(fig, "tests/expected_slice_histogram", compare_images)


# @pytest.mark.skip(reason="temporarily disable")
def test_trace_dial(stan: Stan, compare_images: bool) -> None:
    """Trace dial plot is generated without error."""
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(default)
    fig: go.Figure = backfillz.plot_trace_dial('mu')
    expect_fig(fig, "tests/expected_trace_dial", compare_images)


# @pytest.mark.skip(reason="temporarily disable")
def test_spiral_stream(stan: Stan, compare_images: bool) -> None:
    """Trace dial plot is generated without error."""
    backfillz = Backfillz(stan.fit)
    backfillz.set_theme(demo_2)
    fig: go.Figure = backfillz.plot_spiral_stream('mu')
    expect_fig(fig, "tests/expected_spiral_stream", compare_images)
