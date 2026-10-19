"""The main-run arm R10-main (2026-10-18; no GPU, no volume). Needs the modal package:
    uv run --no-project --with modal --with numpy --with pytest python -m pytest test_main_arm.py -q"""
import pytest

MA = pytest.importorskip("modal_app")


def test_r10_main_is_the_r8s_en3_settings():
    arm = MA.R3_ARMS["R10-main"]
    assert arm == MA.R3_ARMS["R8-s-en3"]
    assert arm is not MA.R3_ARMS["R8-s-en3"]  # its own entry: editing one arm never edits the other


def test_r10_main_differs_from_r8s_by_exactly_the_english_levers():
    arm, base = MA.R3_ARMS["R10-main"], MA.R3_ARMS["R8-s"]
    diff = {k: (base.get(k), arm.get(k)) for k in set(arm) | set(base) if arm.get(k) != base.get(k)}
    assert diff == {"en_teacher": (None, "w3soup_a0.5@0"), "en_kl_scale": (None, 3.0), "wf_en_weight": (None, 3.0)}, diff


def test_r10_main_recipe_is_the_standard_stock_start():
    r = MA.R3_ARMS["R10-main"]["recipe"]
    assert r == dict(mix="v4", init_from="ultra", anchor_init=True, dur_weight=0.0, warmup=20,
                     langs=",".join(MA.RUN2_LANGS))
    arm = MA.R3_ARMS["R10-main"]
    assert arm["lr"] == 1e-4 and arm["l2sp_c"] == 10.0
    assert not arm.get("kl_mask_space") and not arm.get("wf_pen_digits_only")  # the damaging / untested switches stay off


def test_main_variant_has_a_six_hour_cap_and_the_same_arguments():
    import inspect
    import re

    assert MA.MAIN_TIMEOUT_S == 6 * 3600
    assert inspect.signature(MA.retrain3_80_main._raw_f_) == inspect.signature(MA.retrain3._raw_f_)
    src = inspect.getsource(MA)
    deco = lambda fn: re.search(r"@app\.function\(([^@]*?)\)\ndef " + fn + r"\(", src, re.S).group(1)  # noqa: E731
    assert "timeout=MAIN_TIMEOUT_S" in deco("retrain3_80_main") and "A100-80GB" in deco("retrain3_80_main")
    assert "timeout=24 * 3600" in deco("retrain3_80") and "timeout=24 * 3600" in deco("retrain3")  # control: unchanged


def test_r10_main_lr05_is_r10_main_at_half_the_peak_lr():
    arm, base = MA.R3_ARMS["R10-main-lr05"], MA.R3_ARMS["R10-main"]
    diff = {k: (base.get(k), arm.get(k)) for k in set(arm) | set(base) if arm.get(k) != base.get(k)}
    assert diff == {"lr": (1e-4, 5e-5)}, diff
    assert arm["recipe"] == base["recipe"] and arm is not base
