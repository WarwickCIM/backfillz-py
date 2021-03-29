"""Test module for backfillz."""

from backfillz.backfillz import as_backfillz
from tests.generate_sample_fit import generate_fit


def test() -> None:
    # "that backfillz object can be created"
    sample_backfillz = as_backfillz(generate_fit(), verbose=False)
    print(sample_backfillz)


if __name__ == '__main__':
    test()
