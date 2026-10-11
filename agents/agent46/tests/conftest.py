"""Test isolation: no test may read or write the real court docket.

evals/scenarios/docket.json holds an adjuster's real convictions. Tests that score the dev split
would otherwise change with that file. Point every test at an empty docket; tests that exercise
the docket set their own EVOLVE_DOCKET_PATH, which overrides this.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_docket(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLVE_DOCKET_PATH", str(tmp_path / "isolated_docket.json"))
