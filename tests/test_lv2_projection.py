from spiff.lv2 import project_to_epoch


def test_project_to_epoch_returns_reference_for_missing_target_epoch() -> None:
    ra, dec = project_to_epoch(10.0, 20.0, 2016.0, 100.0, -50.0, None)
    assert ra == 10.0
    assert dec == 20.0


def test_project_to_epoch_handles_zero_motion() -> None:
    ra, dec = project_to_epoch(10.0, 20.0, 2016.0, 0.0, 0.0, 59000.0)
    assert abs(ra - 10.0) < 1e-9
    assert abs(dec - 20.0) < 1e-9
