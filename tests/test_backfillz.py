"""Test module for backfillz."""

from tests.generate_sample_fit import generate_fit, Stan

from backfillz.Backfillz import as_backfillz


def test() -> None:
    """Backfillz object can be created."""
    stan = generate_fit()
    file = "expected_backfillz"
    stan.save(file)
    as_backfillz(stan.fit, verbose=False)
    expected_stan = Stan.load(file)
    print(str(stan.fit))
    assert expected_stan.equal(stan)


if __name__ == '__main__':
    test()
