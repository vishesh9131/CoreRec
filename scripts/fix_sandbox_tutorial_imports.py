#!/usr/bin/env python3
"""Fix sandbox tutorial imports to use corerec.sandbox.* paths."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = ROOT / "docs" / "source" / "tutorials"

# Production tutorials — leave imports unchanged
PRODUCTION_STEMS = {
    "dcn_tutorial",
    "deepfm_tutorial",
    "gnnrec_tutorial",
    "mind_tutorial",
    "nasrec_tutorial",
    "sasrec_tutorial",
    "two_tower_tutorial",
    "bert4rec_tutorial",
    "sar_tutorial",
    "ncf_tutorial",
    "fast_tutorial",
    "fast_recommender_tutorial",
    "lightgcn_tutorial",
    "tfidf_tutorial",
    "pipeline_tutorial",
    "index",
}

# Explicit sandbox import mappings (module, class, alias)
SANDBOX_IMPORTS: dict[str, tuple[str, str, str | None]] = {
    "afm": ("corerec.sandbox.collaborative_full.nn_base.AFM_base", "AFM_base", "AFM"),
    "autofi": ("corerec.sandbox.collaborative_full.nn_base.AutoFI_base", "AutoFI_base", "AutoFI"),
    "autoint": ("corerec.sandbox.collaborative_full.nn_base.AutoInt_base", "AutoInt_base", "AutoInt"),
    "bpr": ("corerec.sandbox.collaborative_full.cornac_bpr", "BPR", None),
    "bprmf": ("corerec.sandbox.collaborative_full.bayesian_method_base.bprmf_base", "BPRMF_base", "BPRMF"),
    "bivae": ("corerec.sandbox.collaborative_full.variational_encoder_base.bivae_base", "BiVAE_base", "BiVAE"),
    "bst": ("corerec.sandbox.collaborative_full.nn_base.BST_base", "BST_base", "BST"),
    "caser": ("corerec.sandbox.collaborative_full.sequential_model_base.caser_base", "Caser_base", "Caser"),
    "dcn_base": ("corerec.sandbox.collaborative_full.nn_base.DCN_base", "DCN_base", None),
    "deepcrossing": ("corerec.sandbox.collaborative_full.nn_base.DeepCrossing_base", "DeepCrossing_base", "DeepCrossing"),
    "deepfm_base": ("corerec.sandbox.collaborative_full.nn_base.DeepFM_base", "DeepFM_base", None),
    "deeprec": ("corerec.sandbox.collaborative_full.nn_base.DeepRec_base", "DeepRec_base", "DeepRec"),
    "dien": ("corerec.sandbox.collaborative_full.nn_base.DIEN_base", "DIEN_base", "DIEN"),
    "difm": ("corerec.sandbox.collaborative_full.nn_base.DIFM_base", "DIFM_base", "DIFM"),
    "din": ("corerec.sandbox.collaborative_full.nn_base.DIN_base", "DIN_base", "DIN"),
    "dlrm": ("corerec.sandbox.collaborative_full.nn_base.DLRM_base", "DLRM_base", "DLRM"),
    "ensfm": ("corerec.sandbox.collaborative_full.nn_base.ENSFM_base", "ENSFM_base", "ENSFM"),
    "escmm": ("corerec.sandbox.collaborative_full.nn_base.ESCMM_base", "ESCMM_base", "ESCMM"),
    "esmm": ("corerec.sandbox.collaborative_full.nn_base.ESMM_base", "ESMM_base", "ESMM"),
    "ffm": ("corerec.sandbox.collaborative_full.nn_base.FFM_base", "FFM_base", "FFM"),
    "fgcnn": ("corerec.sandbox.collaborative_full.nn_base.FGCNN_base", "FGCNN_base", "FGCNN"),
    "fibinet": ("corerec.sandbox.collaborative_full.nn_base.Fibinet_base", "Fibinet_base", "Fibinet"),
    "flen": ("corerec.sandbox.collaborative_full.nn_base.FLEN_base", "FLEN_base", "FLEN"),
    "fm": ("corerec.sandbox.collaborative_full.nn_base.FM_base", "FM_base", "FM"),
    "fm_base": ("corerec.sandbox.collaborative_full.nn_base.FM_base", "FM_base", None),
    "gan": ("corerec.sandbox.collaborative_full.nn_base.gan_ufilter_base", "GAN_ufilter_base", "GAN"),
    "gatenet": ("corerec.sandbox.collaborative_full.nn_base.GateNet_base", "GateNet_base", "GateNet"),
    "geoimc": ("corerec.sandbox.collaborative_full.graph_based_base.geoimc", "GeoIMC", None),
    "gnn_base": ("corerec.sandbox.collaborative_full.graph_based_base.GNN_base", "GNN_base", None),
    "gru_cf": ("corerec.sandbox.collaborative_full.nn_base.gru_ufilter_base", "GRU_ufilter_base", "GRU_CF"),
    "lightgcn_base": ("corerec.sandbox.collaborative_full.graph_based_base.lightgcn_base", "LightGCN_base", None),
    "matrixfactorization": ("corerec.sandbox.collaborative_full.mf_base.matrix_factorization_base", "MatrixFactorization_base", "MatrixFactorization"),
    "mf_base": ("corerec.sandbox.collaborative_full.mf_base.mf_base", "MF_base", None),
    "mind_content": ("corerec.sandbox.content_based_full.nn_based_algorithms.mind", "MIND", None),
    "mmoe": ("corerec.sandbox.collaborative_full.nn_base.MMoE_base", "MMoE_base", "MMoE"),
    "monolith": ("corerec.sandbox.collaborative_full.nn_base.Monolith_base", "Monolith_base", "Monolith"),
    "nextitnet": ("corerec.sandbox.collaborative_full.sequential_model_base.nextitnet_base", "NextItNet_base", "NextItNet"),
    "nfm": ("corerec.sandbox.collaborative_full.nn_base.NFM_base", "NFM_base", "NFM"),
    "ple": ("corerec.sandbox.collaborative_full.nn_base.PLE_base", "PLE_base", "PLE"),
    "pnn": ("corerec.sandbox.collaborative_full.nn_base.PNN_base", "PNN_base", "PNN"),
    "rbm": ("corerec.sandbox.collaborative_full.rbm", "RBM", None),
    "rlrmc": ("corerec.sandbox.collaborative_full.rlrmc", "RLRMC", None),
    "slirec": ("corerec.sandbox.collaborative_full.sli", "SLI", "SLIRec"),
    "sum": ("corerec.sandbox.collaborative_full.sum", "SUM", None),
    "svd": ("corerec.sandbox.collaborative_full.mf_base.svd_base", "SVD_base", "SVD"),
    "tdm": ("corerec.sandbox.collaborative_full.nn_base.TDM_base", "TDM_base", "TDM"),
    "userbased": ("corerec.sandbox.collaborative_full.mf_base.user_based_base", "UserBased_base", "UserBased"),
    "vmf": ("corerec.sandbox.collaborative_full.mf_base.vmf_base", "VMF_base", "VMF"),
    "widedeep": ("corerec.sandbox.collaborative_full.nn_base.WideDeep_base", "WideDeep_base", "WideDeep"),
    "youtubednn": ("corerec.engines.content_based.nn_based_algorithms.Youtube_dnn", "YoutubeDNN", None),
    "a2svd": ("corerec.sandbox.collaborative_full.mf_base.a2svd_base", "A2SVD_base", "A2SVD"),
    "als": ("corerec.sandbox.collaborative_full.mf_base.als_base", "ALS_base", "ALS"),
}

IMPORT_LINE = re.compile(
    r"^from corerec\.engines(?:\.[\w.]+)? import (.+)$",
    re.MULTILINE,
)


def stem_from_path(path: Path) -> str:
    name = path.stem  # e.g. afm_tutorial
    return name.replace("_tutorial", "")


def build_import_line(module: str, cls: str, alias: str | None) -> str:
    if alias and alias != cls:
        return f"from {module} import {cls} as {alias}"
    return f"from {module} import {cls}"


def fix_tutorial(path: Path) -> bool:
    stem = stem_from_path(path)
    if path.stem in PRODUCTION_STEMS or stem not in SANDBOX_IMPORTS:
        return False

    text = path.read_text(encoding="utf-8")
    module, cls, alias = SANDBOX_IMPORTS[stem]
    new_import = build_import_line(module, cls, alias)

    def repl(match: re.Match) -> str:
        return new_import

    new_text, n = IMPORT_LINE.subn(repl, text, count=1)
    if n == 0:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> None:
    fixed = 0
    for path in sorted(TUTORIAL_DIR.glob("*_tutorial.md")):
        if fix_tutorial(path):
            fixed += 1
            print(f"fixed: {path.name}")
    print(f"Done. Fixed {fixed} tutorials.")


if __name__ == "__main__":
    main()
