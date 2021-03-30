"""Test module for backfillz."""

from tests.generate_sample_fit import generate_fit

from backfillz.Backfillz import as_backfillz


def test() -> None:
    """Backfillz object can be created."""
    sample_backfillz = as_backfillz(generate_fit(), verbose=False)
    print(sample_backfillz)


if __name__ == '__main__':
    test()
