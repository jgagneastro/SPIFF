from pathlib import Path

from spiff.examples import SIMP_J0136_TARGET, build_simp_j0136_test_commands, simp_j0136_target_dir


def test_simp_j0136_target_dir_matches_output_naming(tmp_path: Path) -> None:
    target_dir = simp_j0136_target_dir(tmp_path)

    assert target_dir == (
        tmp_path.resolve() / "SIMP-J013656.5093347.3_RA24.241250_DEC9.563070"
    )


def test_simp_j0136_commands_include_verified_parameters(tmp_path: Path) -> None:
    commands = build_simp_j0136_test_commands(tmp_path)

    assert len(commands) == 3
    assert f"--ra {SIMP_J0136_TARGET['ra_deg']}" in commands[0]
    assert f"--reference-pmra-masyr {SIMP_J0136_TARGET['reference_pmra_masyr']}" in commands[0]
    assert f"--reference-pmdec-masyr {SIMP_J0136_TARGET['reference_pmdec_masyr']}" in commands[0]
    assert str(simp_j0136_target_dir(tmp_path) / "compiled_results.csv") in commands[2]
