"""Dry build of the p3 / window parts of the mix with and without the realx1 extra pass (CPU, new directories only).

Runs mix_build.build4 for the given languages into <mix_dir>/ (a dry directory: the real mixes are never touched) with the
build's balance / dedupe / edge-fix / win15 settings, so the effect of an extra pass on the selected real rows, the class
census and the win15 windows is measured by the build itself:

    modal run realx1_dry.py::run --langs es --mix-dir mix4dryrx_ctl                       # control: l1conv extra pass only
    modal run realx1_dry.py::run --langs es --mix-dir mix4dryrx_new --realx1 --edge-fix ml/edgefix2x \\
        --win15-dir ml/win15b,ml/win15sym,ml/realx1wb
"""

import json
import os

import modal

VOL = "/vol"
vol = modal.Volume.from_name("parakeet-spoken-form")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("ffmpeg")
         .uv_pip_install("numpy", "scipy", "soundfile", "librosa")
         .add_local_python_source("common", "mix_build", "census_common", "ml_cands", "ml_balance", "ml_p3_core",
                                 "score_retrain", "labeller", "readings", "compound_split", "win15", "edge_fix",
                                 "alnum_relabel")
         .add_local_file("ml_grammar_table.json", "/root/ml_grammar_table.json"))
app = modal.App("parakeet-realx1-dry")


@app.function(image=image, volumes={VOL: vol}, cpu=8, memory=32768, timeout=3 * 3600)
def dry(langs: str, mix_dir: str, extra: str, edge_fix: str, win15_dir: str, win15_share: float) -> dict:
    import mix_build

    assert mix_dir.startswith("mix4dry"), "a dry build writes a mix4dry* directory"
    vol.reload()
    assert not os.path.exists(f"{VOL}/{mix_dir}"), f"{mix_dir} exists: use a new directory"
    out = mix_build.build4(VOL, tuple(langs.split(",")), rows_h=99.0, win_h=2.9, dedupe=True, seam_h=5.0, plain_share=0.6,
                           balance=True, mix_dir=mix_dir, seam_long_h=6.0, split_langs=("de", "nl", "it-"),
                           win15_share=win15_share, win15_dir=win15_dir, edge_fix_dir=edge_fix,
                           extra_passes=tuple(p for p in extra.split(",") if p),
                           relabel_rows="ml/alnumA/{lang}/rows_alnumA.jsonl", relabel_parts_dir="mix4final8_alnumA",
                           relabel_tts=True)
    json.dump(out, open(f"{VOL}/{mix_dir}/dry_report.json", "w"), indent=1, ensure_ascii=False, default=str)
    vol.commit()
    return {l: {k: v for k, v in s.items() if not isinstance(v, (dict, list))} for l, s in out.get("langs", {}).items()}


@app.local_entrypoint()
def run(langs: str = "es", mix_dir: str = "mix4dryrx_ctl", realx1: bool = False, edge_fix: str = "ml/edgefix2",
        win15_dir: str = "ml/win15b", win15_share: float = 0.11):
    extra = "ml/l1conv/{lang}/rows_l1conv.jsonl" + (",ml/realx1/{lang}/rows_realx1_regional.jsonl" if realx1 else "")
    print(json.dumps(dry.remote(langs, mix_dir, extra, edge_fix, win15_dir, win15_share), indent=1))
