"""Test module for backfillz."""

from tests.generate_sample_fit import Stan, generate_fit
from backfillz.Backfillz import as_backfillz


def test() -> None:
    """Backfillz object can be created."""
    stan = generate_fit()
    file = "model_fit"
    stan.save(file)
    sample_backfillz = as_backfillz(stan.fit, verbose=False)
    expected_stan = Stan.load(file)
    assert expected_stan.equal(stan)

    print(sample_backfillz)


if __name__ == '__main__':
    test()
