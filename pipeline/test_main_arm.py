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
