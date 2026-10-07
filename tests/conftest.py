import pytest

from eta_eats.data import prepare
from eta_eats.synthetic import generate, write_synthetic


@pytest.fixture(scope="session")
def raw():
    return generate(2500, seed=5)


@pytest.fixture(scope="session")
def frame(raw):
    prepared, _ = prepare(raw)
    return prepared


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    write_synthetic(out, n_train=2000, n_test=300, seed=7)
    return out
