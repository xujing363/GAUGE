import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

common.configure_page("About & Model Card")
bundle = common.sidebar_mode_selector()

st.title("ℹ️ About & Model Card")

st.markdown(
    """
## What is GAUGE?

**GAUGE** (*state-adaptive knowledge-graph gating*) is a computational model that predicts a
tumour's response to a drug from three inputs: a bulk tumour transcriptome (cell line **or**
primary patient sample), the drug's chemical structure, and three structured pharmacological
knowledge graphs (**ChEMBL**, mechanism of action; **DRKG**, gene/protein drug-target
interactions; **PrimeKG**, protein–disease associations).

Rather than merging the three knowledge graphs into one fixed structure, GAUGE learns, for
every tumour–drug pair, a context-conditioned attention weight over each source plus a gate
that controls how strongly that knowledge is injected into the drug representation. The same
single, frozen model — trained only on cell-line screening data — is applied without
retraining to cell-line generalisation, zero-shot patient stratification, single-cell
vulnerability scoring, drug-combination prioritisation, and molecular-design guidance.
"""
)

with st.expander("Architecture summary"):
    st.markdown(
        """
- **Transcriptome encoder**: 2-layer MLP, top-2000-variance genes (selected on training cell
  lines) → 128-d tumour-state latent.
- **Chemical encoder**: 2-layer MLP, 2048-bit Morgan fingerprint (radius 2) → 128-d latent.
- **Knowledge-graph branches**: 3 independent 2-layer relational graph-attention networks
  (one per source), drug nodes initialised from the chemical latent.
- **Source attention**: softmax over the 3 branches, conditioned on tumour state, chemical
  latent, branch embedding, KG coverage mask, and log-degree.
- **Injection gate**: `z_a = z_chem + sigmoid(W[z_s, z_chem, z_KG]) · z_KG`.
- **Interaction + heads**: tumour–drug interaction module (`z_s`, `z_a`, `z_s ⊙ z_a`) feeding
  an absolute-AUC head and a sigmoid-bounded relative-sensitive-value head trained by
  drug-level grouped ranking.
- **Inputs explicitly *not* used**: mutation calls, copy-number variation, tumour stage, age,
  cancer-type label, survival outcome, pathology, or curated pathway gene sets — GAUGE only
  ever sees expression + chemistry + the three KGs.
"""
    )

st.subheader("Reported performance (this checkpoint)")
metrics = pd.DataFrame(bundle.meta.get("reported_metrics_this_seed", []))
if not metrics.empty:
    st.dataframe(metrics, use_container_width=True)
st.caption(
    f"Split type: {bundle.meta.get('split_type')}. "
    "Single-seed snapshot of the published benchmark; the paper reports the mean ± s.e.m. across 5 seeds."
)

st.subheader("Intended use")
st.markdown(
    """
GAUGE is intended for **computational hypothesis generation and prioritisation**: ranking
candidate drugs for a profiled sample, flagging knowledge-graph-supported mechanistic
rationale, and exploring combination or design-guidance prioritisation. It is a **research
tool**.
"""
)

st.subheader("Limitations")
st.markdown(
    """
- Trained exclusively on in vitro cell-line screens (GDSC or, in PRISM mode, the Broad
  Repurposing Hub secondary screen); patient-level predictions are **zero-shot transfers**,
  not predictions calibrated on patient outcome data.
- Combination scores are derived post hoc from two single-agent predictions — GAUGE has never
  seen combination-response labels.
- Knowledge-graph explainability in this app is **source-level** (which of the 3 graphs
  contributed), not edge- or path-level mechanistic tracing.
- Predictions for chemistry outside the bundled library fall back to chemistry-only reasoning
  (no knowledge-graph evidence is available for genuinely novel compounds).
- This tool does **not** replace clinical judgement, regulatory-approved companion
  diagnostics, or a treating physician's assessment.
"""
)

st.subheader("Citation")
st.code(
    "Xu J, Chen L. GAUGE: Cancer drug-response prediction by state-adaptive "
    "knowledge-graph gating. Manuscript in preparation.",
    language="text",
)

st.subheader("Software")
st.markdown(
    """
- **Version**: see `gauge_core.__version__` / the repository `CHANGELOG`.
- **License**: see `LICENSE` in the software root.
- **Source data dependencies**: GDSC1/2, PRISM (Broad Repurposing Hub) + DepMap, ChEMBL,
  DRKG, PrimeKG (see `docs/DATA_SOURCES.md`).
- This app performs no model retraining; it loads the exact published benchmark checkpoints.
"""
)
