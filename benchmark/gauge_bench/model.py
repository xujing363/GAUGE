"""
GAUGE: Cancer drug-response prediction by state-adaptive knowledge-graph gating.

Reference implementation of the model described in the GAUGE manuscript
(Methods: "Model architecture", "Relation Graph Attention Layer",
"State-adaptive knowledge-graph gating module", "Tumour-Drug interacting and
predicting network", "Relative-sensitive-value target and prediction",
"Training objectives").

Three-stage pipeline
  1. Encoders          x_s (2000 HVG + 3 cell stats) -> 256 -> z_s (128)
                       x_fp (Morgan r=2, 2048 bit)   -> 512 -> z_chem (128)
                       per-KG-source node states -> 2 relation-aware GAT layers
  2. State-adaptive KG gating
                       context source attention (386 -> 128 -> 1) over the three
                       branches (ChEMBL / DRKG / PrimeKG) -> z_kg
                       gate g = sigmoid(W[z_s, z_chem, z_kg]) -> z_a = z_chem + g*z_kg
  3. Interaction + heads
                       b = T([z_s, z_a, z_s*z_a])  (384 -> 256 -> 128)
                       absolute-AUC head, relative-sensitive-value (RTV) head,
                       tumour(cell)-residual head

Everything needed to build targets, run a forward pass and compute the active
training loss is contained in this file. Only numpy / pandas / torch are required
(RDKit only for the optional fingerprint helper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

LATENT_DIM = 128
FP_DIM = 2048
BRANCH_NAMES = ("ChEMBL", "DRKG", "PrimeKG")


# ---------------------------------------------------------------------------
# 1. Data preprocessing / targets  (Methods: Data preprocessing, RTV target)
# ---------------------------------------------------------------------------


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = FP_DIM) -> np.ndarray:
    """Morgan fingerprint, radius 2, 2048 bits (Methods: Drug preprocessing)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)


def canonical_group_key(row: pd.Series) -> str:
    """Deterministic, response-independent drug identity key (InChIKey preferred)."""
    for field_name in ("inchikey", "canonical_smiles", "smiles", "DRUG_NAME"):
        value = str(row.get(field_name, "")).strip()
        if value and value.lower() not in {"nan", "<na>", ""}:
            return value
    return f"drug_id::{int(row['DRUG_ID'])}"


def build_canonical_drug_table(drug_table: pd.DataFrame) -> pd.DataFrame:
    """Index drugs by canonical group rather than raw DRUG_ID order.

    Splits are defined at the canonical-drug level so that duplicate
    representations of the same chemical entity cannot cross partitions.
    """
    out = drug_table.copy()
    out["canonical_group_key"] = out.apply(canonical_group_key, axis=1)
    codes, _ = pd.factorize(out["canonical_group_key"], sort=True)
    out["canonical_drug_index"] = codes.astype(np.int64)
    return out


def build_relative_value_target(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    response_col: str = "AUC",
    drug_col: str = "DRUG_ID",
) -> pd.DataFrame:
    """Within-drug relative-sensitive-value (RTV) target, eq. (21).

        RTV_{i,d} = 1 - rank_d(AUC_i) / N_d

    Lower AUC (greater sensitivity) -> higher RTV. The reference distribution is
    fitted on TRAINING cell lines only and then frozen.
    """
    out = frame.copy()
    out["relative_value_train"] = np.nan
    train_mask = out[split_col].eq("train").to_numpy()
    if not train_mask.any():
        return out
    reference = {
        int(d): np.sort(g[response_col].astype(np.float32).to_numpy())
        for d, g in out.loc[train_mask].groupby(drug_col, observed=True)
    }
    drug_ids = out[drug_col].astype(int).to_numpy()
    response = out[response_col].astype(np.float32).to_numpy()
    rtv = np.full((len(out),), np.nan, dtype=np.float32)
    for drug_id, sorted_auc in reference.items():
        m = drug_ids == drug_id
        rtv[m] = 1.0 - np.searchsorted(sorted_auc, response[m], side="right") / len(sorted_auc)
    out.loc[train_mask, "relative_value_train"] = rtv[train_mask]
    out["relative_value_eval"] = rtv
    return out


def build_cell_residual_target(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    response_col: str = "AUC",
    cell_col: str = "SANGER_MODEL_ID",
) -> pd.DataFrame:
    """Tumour-normalised residual response: AUC minus the cell's train-median AUC."""
    out = frame.copy()
    out["cell_residual_auc_train"] = np.nan
    train_mask = out[split_col].eq("train")
    if not train_mask.any():
        return out
    baseline = (
        out.loc[train_mask].groupby(cell_col, observed=True)[response_col].median().astype(float)
    )
    out["cell_train_baseline"] = out[cell_col].map(baseline).astype(float)
    residual = out[response_col].astype(float) - out["cell_train_baseline"]
    out.loc[train_mask, "cell_residual_auc_train"] = residual.loc[train_mask]
    out["cell_residual_auc_eval"] = residual
    return out


def build_cell_train_statistics(
    frame: pd.DataFrame,
    *,
    split_col: str = "split",
    response_col: str = "AUC",
    cell_col: str = "SANGER_MODEL_ID",
) -> pd.DataFrame:
    """Three train-only cell statistics appended to the 2000-HVG state vector."""
    train_rows = frame.loc[frame[split_col].eq("train"), [cell_col, response_col]]
    group = train_rows.groupby(cell_col, observed=True)[response_col]
    stats = pd.DataFrame(
        {
            "cell_auc_train_mean": group.mean().astype(np.float32),
            "cell_auc_train_median": group.median().astype(np.float32),
        }
    )
    global_median = float(train_rows[response_col].median())
    stats["cell_centered_sensitivity_train"] = (
        global_median - stats["cell_auc_train_median"]
    ).astype(np.float32)
    return stats


# ---------------------------------------------------------------------------
# 2. Encoders  (Methods eq. 1-4)
# ---------------------------------------------------------------------------


class StateEncoder(nn.Module):
    """x_s -> 256 -> z_s (128). Two-layer MLP, ReLU."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DrugEncoder(nn.Module):
    """2048-bit Morgan fingerprint -> 512 -> z_chem (128). Two-layer MLP, ReLU."""

    def __init__(self, fp_dim: int = FP_DIM, hidden_dim: int = 512, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net(fp)


# ---------------------------------------------------------------------------
# 3. Relation-aware graph attention  (Methods eq. 5-7)
# ---------------------------------------------------------------------------


class RelationGraphAttentionLayer(nn.Module):
    """For edge u -> v: message = W_msg h_u + r_{type}; sigmoid attention on
    [h_v || message]; mean aggregation over in-edges; residual + LayerNorm."""

    def __init__(self, latent_dim: int, n_relations: int, dropout: float = 0.1):
        super().__init__()
        self.msg = nn.Linear(latent_dim, latent_dim, bias=False)
        self.rel = nn.Embedding(max(n_relations, 1), latent_dim)
        self.att = nn.Linear(latent_dim * 2, 1)
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_dropout: float = 0.0,
        collect_scores: list | None = None,
    ) -> torch.Tensor:
        # collect_scores, when given, receives (src, dst, edge_type, score) for
        # every edge this layer actually used.  Read-only instrumentation for
        # the per-edge attention panel; it does not alter the computation.
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index[0], edge_index[1]
        if self.training and edge_dropout > 0:
            # One nonzero() shared by the three gathers: boolean indexing would
            # run its own nonzero() per tensor. Draw order is unchanged, so the
            # RNG stream (and therefore the dropout masks below) is identical.
            keep = torch.nonzero(
                torch.rand(src.shape, device=h.device) >= float(edge_dropout), as_tuple=False
            ).squeeze(1)
            src = src.index_select(0, keep)
            dst = dst.index_select(0, keep)
            edge_type = edge_type.index_select(0, keep)
            if src.numel() == 0:
                return h
        # W_msg h_u is projected on the N nodes and only then gathered to the E
        # edges: identical rows, but the matmul runs on ~12.5k rows instead of
        # ~78k. Likewise the attention linear is split over its two input
        # blocks, W_att [h_v || m] = W_v h_v + W_m m + b, which lets W_v h_v be
        # computed per node and removes the [E, 2*latent] concat entirely.
        msg = self.msg(h).index_select(0, src) + self.rel(edge_type.clamp_min(0))
        d = h.shape[1]
        att_v = (h @ self.att.weight[:, :d].t()).index_select(0, dst)
        score = torch.sigmoid(att_v + msg @ self.att.weight[:, d:].t() + self.att.bias)
        if collect_scores is not None:
            collect_scores.append((src.detach().cpu(), dst.detach().cpu(),
                                   edge_type.detach().cpu(), score.detach().squeeze(-1).cpu()))
        msg = self.dropout(msg * score)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        # In-degree by bincount: an exact integer count, one kernel instead of
        # allocating a ones vector and scattering it.
        deg = torch.bincount(dst, minlength=h.shape[0]).to(h.dtype).unsqueeze(1)
        agg = agg / deg.clamp_min(1.0)
        return self.norm(h + F.relu(agg))


# ---------------------------------------------------------------------------
# 4. Multi-source KG encoder + context-conditioned source attention (eq. 8-11)
# ---------------------------------------------------------------------------


@dataclass
class KGArtifacts:
    """Pre-built multi-source KG prior.

    node_table : columns [node_id, node_type]                 (global node index)
    edge_table : columns [src, dst, edge_type, source]        (global node index)
    coverage   : indexed rows per canonical drug, columns
                 has_{S}, {S}_node_id, source_weight_{S}, graph_degree_{S}
    drug_ids   : canonical drug ids, in bank order
    """

    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    coverage: pd.DataFrame
    drug_ids: list[int]
    branch_names: tuple[str, ...] = BRANCH_NAMES

    @property
    def drug_to_local(self) -> dict[int, int]:
        return {int(d): i for i, d in enumerate(self.drug_ids)}


class MultiKGActionEncoder(nn.Module):
    """Per-source relation-aware GAT branches + context-conditioned source attention.

    Node states are initialised from node-identity + node-type embeddings; drug
    nodes are overwritten by a projection of z_chem, coupling chemical structure
    to graph propagation. Each branch applies two RelationGraphAttentionLayers.
    """

    def __init__(
        self,
        kg: KGArtifacts,
        drug_fingerprint_bank: np.ndarray,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()
        self.branch_names = list(kg.branch_names)
        self.n_branches = len(self.branch_names)
        self.latent_dim = latent_dim
        self.drug_ids = [int(x) for x in kg.drug_ids]
        self.drug_to_local = kg.drug_to_local

        # --- node embeddings -------------------------------------------------
        self.n_nodes = int(len(kg.node_table))
        node_types = kg.node_table.get("node_type", pd.Series(["node"] * self.n_nodes)).astype(str).tolist()
        type_vocab = {name: i for i, name in enumerate(sorted(set(node_types) | {"node"}))}
        self.register_buffer(
            "node_type_ids",
            torch.as_tensor([type_vocab.get(t, 0) for t in node_types], dtype=torch.long),
            persistent=False,
        )
        self.node_id_embedding = nn.Embedding(max(self.n_nodes, 1), latent_dim)
        self.node_type_embedding = nn.Embedding(max(len(type_vocab), 1), latent_dim)
        self.chem_to_node = nn.Linear(latent_dim, latent_dim)
        self.register_buffer(
            "drug_fingerprint_bank",
            torch.as_tensor(np.asarray(drug_fingerprint_bank, dtype=np.float32)),
            persistent=False,
        )

        # --- per-drug coverage / node id / source weight / degree -------------
        n_drugs = len(self.drug_ids)
        node_ids = np.zeros((n_drugs, self.n_branches), dtype=np.int64)
        mask = np.zeros((n_drugs, self.n_branches), dtype=np.float32)
        weight = np.zeros((n_drugs, self.n_branches), dtype=np.float32)
        degree = np.zeros((n_drugs, self.n_branches), dtype=np.float32)
        cov = kg.coverage.set_index("DRUG_ID") if "DRUG_ID" in kg.coverage.columns else kg.coverage
        for i, drug_id in enumerate(self.drug_ids):
            if drug_id not in cov.index:
                continue
            row = cov.loc[drug_id]
            for j, branch in enumerate(self.branch_names):
                node_ids[i, j] = int(row.get(f"{branch}_node_id", 0))
                mask[i, j] = float(row.get(f"has_{branch}", 0.0))
                weight[i, j] = float(row.get(f"source_weight_{branch}", mask[i, j]))
                degree[i, j] = float(row.get(f"graph_degree_{branch}", 0.0))
        for name, arr, dtype in (
            ("branch_node_ids", node_ids, torch.long),
            ("branch_mask", mask, torch.float32),
            ("branch_weight", weight, torch.float32),
            ("branch_degree", degree, torch.float32),
        ):
            self.register_buffer(name, torch.as_tensor(arr, dtype=dtype), persistent=False)

        # --- per-source edge buffers + two GAT layers -------------------------
        self.branches = nn.ModuleList()
        self._edge_keys: list[tuple[str, str]] = []
        for branch in self.branch_names:
            sub = kg.edge_table.loc[kg.edge_table["source"].astype(str).eq(branch)]
            rels = sorted(sub["edge_type"].astype(str).unique().tolist()) if len(sub) else ["self"]
            rel_to_id = {name: i for i, name in enumerate(rels)}
            if len(sub):
                edge_index = torch.as_tensor(sub[["src", "dst"]].astype(int).to_numpy().T, dtype=torch.long)
                edge_type = torch.as_tensor([rel_to_id[t] for t in sub["edge_type"].astype(str)], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_type = torch.empty((0,), dtype=torch.long)
            ik, tk = f"edge_index_{branch.lower()}", f"edge_type_{branch.lower()}"
            self.register_buffer(ik, edge_index, persistent=False)
            self.register_buffer(tk, edge_type, persistent=False)
            self._edge_keys.append((ik, tk))
            self.branches.append(
                nn.ModuleList(
                    [
                        RelationGraphAttentionLayer(latent_dim, len(rels)),
                        RelationGraphAttentionLayer(latent_dim, len(rels)),
                    ]
                )
            )

        # --- context-conditioned source attention: 386 -> 128 -> 1 -----------
        # input = [z_s (128) | z_chem (128) | branch embedding (128) | mask (1) | log1p(degree) (1)]
        self.alpha = nn.Sequential(
            nn.Linear(latent_dim * 3 + 2, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1)
        )
        # non-adaptive fusion ablation (concat_only)
        self.concat_projection = nn.Linear(latent_dim * self.n_branches, latent_dim)
        # "Fixed weights" ablation: one global logit per source, no state
        # conditioning.  Unused unless kg_mode == "fixed_source_weights", so its
        # presence does not change any other configuration's computation.
        self.fixed_source_logits = nn.Parameter(torch.zeros(self.n_branches))

    # -- helpers ------------------------------------------------------------
    def local_indices(self, drug_ids: Sequence[int], device: Any = None) -> torch.Tensor:
        return torch.as_tensor(
            [self.drug_to_local.get(int(x), 0) for x in drug_ids], dtype=torch.long, device=device
        )

    def _initial_nodes(self, drug_latent_bank: torch.Tensor) -> torch.Tensor:
        # node_id_embedding(arange(n_nodes)) is exactly its weight matrix.
        h = self.node_id_embedding.weight + self.node_type_embedding(self.node_type_ids)
        drug_init = self.chem_to_node(drug_latent_bank)
        # The three sources hold disjoint node blocks, so every drug node id is
        # distinct and all branches can be written in a single index_copy.
        return h.index_copy(
            0,
            self.branch_node_ids.t().reshape(-1),
            drug_init.repeat(self.n_branches, 1),
        )

    def branch_embeddings(
        self, drug_latent_bank: torch.Tensor, edge_dropout: float = 0.0,
        collect_scores: dict | None = None,
    ) -> torch.Tensor:
        """-> (n_drugs, n_branches, latent_dim); the per-source drug-node states.

        collect_scores, when given, is filled with {branch_name: [(src, dst,
        edge_type, score), ...]} -- one entry per graph-attention layer.
        """
        h0 = self._initial_nodes(drug_latent_bank)
        outs = []
        for j, layers in enumerate(self.branches):
            ik, tk = self._edge_keys[j]
            h = h0
            bucket = [] if collect_scores is not None else None
            for layer in layers:
                h = layer(h, getattr(self, ik), getattr(self, tk),
                          edge_dropout=edge_dropout, collect_scores=bucket)
            if collect_scores is not None:
                collect_scores[self.branch_names[j]] = bucket
            outs.append(h.index_select(0, self.branch_node_ids[:, j]))
        return torch.stack(outs, dim=1)

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        z_s: torch.Tensor,
        z_chem: torch.Tensor,
        drug_idx: torch.Tensor,
        drug_latent_bank: torch.Tensor,
        *,
        mode: str = "multikg_gat",
        edge_dropout: float = 0.0,
        branch_all: torch.Tensor | None = None,
        branch_mask_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if branch_all is None:
            branch_all = self.branch_embeddings(drug_latent_bank, edge_dropout=edge_dropout)
        z_branch = branch_all.index_select(0, drug_idx)
        mask = self.branch_weight.index_select(0, drug_idx).clone()
        degree = self.branch_degree.index_select(0, drug_idx)
        if branch_mask_override is not None:
            # per-row, per-source keep/drop.  Multiplicative, so it can only
            # take coverage away, never invent a branch the drug does not have.
            mask = mask * branch_mask_override.to(mask.dtype)

        # ----- fusion / gating variants --------------------------------------
        if mode in {"chembl_only", "drkg_only", "primekg_only"}:
            keep = {"chembl_only": 0, "drkg_only": 1, "primekg_only": 2}[mode]
            sel = torch.zeros_like(mask)
            sel[:, keep] = 1.0
            mask = mask * sel
        elif mode == "kg_masked":            # test-drug KG masking at inference
            mask = torch.zeros_like(mask)
        elif mode == "shuffled_prior" and z_branch.shape[0] > 1:
            z_branch = z_branch.index_select(0, torch.randperm(z_branch.shape[0], device=z_branch.device))
        elif mode == "random_prior":
            z_branch = torch.randn_like(z_branch)

        if mode == "uniform_avg":            # "average" bar: uniform source weights
            alpha = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return {
                "z_kg": (z_branch * alpha.unsqueeze(-1)).sum(dim=1),
                "kg_alpha": alpha,
                "kg_mask": mask,
                "kg_degree": degree,
                "z_branch": z_branch,
            }

        if mode == "fixed_source_weights":   # "Fixed weights": learned but NOT
            # state-conditioned -- one global logit per source, softmax over the
            # available branches.  Retrained (the parameter has to be learned).
            logits = self.fixed_source_logits.unsqueeze(0).expand(z_branch.shape[0], -1)
            alpha = torch.softmax(logits.masked_fill(mask <= 0, float("-inf")), dim=1)
            alpha = torch.nan_to_num(alpha) * mask
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return {
                "z_kg": (z_branch * alpha.unsqueeze(-1)).sum(dim=1),
                "kg_alpha": alpha,
                "kg_mask": mask,
                "kg_degree": degree,
                "z_branch": z_branch,
            }

        if mode == "concat_only":            # non-adaptive fusion ablation
            flat = (z_branch * mask.unsqueeze(-1)).reshape(z_branch.shape[0], -1)
            alpha = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return {
                "z_kg": self.concat_projection(flat),
                "kg_alpha": alpha,
                "kg_mask": mask,
                "kg_degree": degree,
                "z_branch": z_branch,
            }

        # ----- context-conditioned source attention, eq. (8)-(11) -------------
        z_s_ctx = torch.zeros_like(z_s) if mode == "no_state_attention" else z_s
        log_deg = torch.log1p(degree)
        # Score all branches in one pass: the MLP is row-wise, so stacking the
        # three [z_s | z_chem | branch | mask | log1p(deg)] rows per sample and
        # folding them into the batch dimension is identical to scoring them
        # one branch at a time, with a third of the kernel launches.
        n_b, ctx = self.n_branches, torch.cat([z_s_ctx, z_chem], dim=1)
        att_in = torch.cat(
            [
                ctx.unsqueeze(1).expand(-1, n_b, -1),
                z_branch,
                mask.unsqueeze(-1),
                log_deg.unsqueeze(-1),
            ],
            dim=2,
        )
        logits = self.alpha(att_in.reshape(-1, att_in.shape[-1])).reshape(-1, n_b)
        logits = logits.masked_fill(mask <= 0, -1e9)          # coverage-aware masking
        alpha = torch.softmax(logits, dim=1)
        has_any = mask.sum(dim=1, keepdim=True) > 0
        alpha = torch.where(has_any, alpha * mask, torch.zeros_like(mask))
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        z_kg = (z_branch * alpha.unsqueeze(-1)).sum(dim=1)     # eq. (11)
        return {
            "z_kg": z_kg,
            "kg_alpha": alpha,
            "kg_mask": mask,
            "kg_degree": degree,
            "z_branch": z_branch,
        }


# ---------------------------------------------------------------------------
# 5. GAUGE  (gating eq. 12-15; interaction eq. 16-18; heads eq. 19-22)
# ---------------------------------------------------------------------------


class GAUGE(nn.Module):
    def __init__(
        self,
        state_dim: int,
        kg: KGArtifacts | None = None,
        drug_fingerprint_bank: np.ndarray | None = None,
        fp_dim: int = FP_DIM,
        latent_dim: int = LATENT_DIM,
        chem_dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        # Pathway dropout on the chemical representation (training only):
        # the whole z_chem vector is zeroed on a random fraction of training
        # rows, so on those rows the response can only be reached through the
        # knowledge-graph branch.  Inactive under model.eval().
        self.chem_dropout = float(chem_dropout)
        self.state_encoder = StateEncoder(state_dim, latent_dim=latent_dim)
        self.drug_encoder = DrugEncoder(fp_dim=fp_dim, latent_dim=latent_dim)
        self.kg_action_encoder = (
            MultiKGActionEncoder(kg, drug_fingerprint_bank, latent_dim=latent_dim)
            if kg is not None and drug_fingerprint_bank is not None
            else None
        )
        # gating network: [z_s | z_chem | z_kg] -> 128 injection gate, eq. (12)-(13)
        self.gate = nn.Linear(latent_dim * 3, latent_dim)
        # tumour-drug interaction encoder, eq. (16)-(18)
        self.terminal = nn.Sequential(
            nn.Linear(latent_dim * 3, 256), nn.ReLU(), nn.Linear(256, latent_dim), nn.ReLU()
        )
        self.raw_auc_head = nn.Linear(latent_dim, 1)             # absolute AUC, eq. (19)
        self.relative_value_head = nn.Linear(latent_dim, 1)      # RTV, eq. (20)/(22)
        self.cell_residual_head = nn.Sequential(                 # tumour-normalised residual
            nn.Linear(latent_dim * 3, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1)
        )

    def _drop_chem(self, z_chem: torch.Tensor) -> torch.Tensor:
        """Zero the whole chemical vector on a random subset of rows (train only).

        Scaled by 1/(1-p) so the expected value is unchanged.
        """
        if not self.training or self.chem_dropout <= 0.0:
            return z_chem
        keep = torch.rand(z_chem.shape[0], 1, device=z_chem.device) >= self.chem_dropout
        return z_chem * keep.to(z_chem.dtype) / (1.0 - self.chem_dropout)

    def precompute_branch_all(self, edge_dropout: float = 0.0) -> torch.Tensor | None:
        """Propagate the KG once per epoch/eval pass (drug bank is batch-independent)."""
        if self.kg_action_encoder is None:
            return None
        bank = self.drug_encoder(self.kg_action_encoder.drug_fingerprint_bank)
        return self.kg_action_encoder.branch_embeddings(bank, edge_dropout=edge_dropout)

    def forward(
        self,
        state: torch.Tensor,
        drug_fp: torch.Tensor | None = None,
        drug_idx: torch.Tensor | None = None,
        *,
        kg_mode: str = "multikg_gat",
        edge_dropout: float = 0.0,
        use_prior: bool = True,
        branch_all: torch.Tensor | None = None,
        fusion_weight: float = 0.0,
        branch_mask_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_s = self.state_encoder(state)

        # Strict title-level controls remove context adaptation from BOTH
        # components of the knowledge-graph-gating module.  The source encoder
        # receives the corresponding non-adaptive source-weighting mode below;
        # the downstream injection gate is replaced after z_kg is constructed.
        source_mode = {
            "strict_fixed_both": "fixed_source_weights",
            "strict_uniform_both": "uniform_avg",
            "reviewer_fixed_no_gate": "fixed_source_weights",
            "reviewer_average_no_gate": "uniform_avg",
            "reviewer_concat_no_gate": "concat_only",
        }.get(kg_mode, kg_mode)

        kg_payload: dict[str, torch.Tensor] = {}
        if use_prior and self.kg_action_encoder is not None and drug_idx is not None:
            bank = self.drug_encoder(self.kg_action_encoder.drug_fingerprint_bank)
            z_chem = bank.index_select(0, drug_idx)
            z_chem = self._drop_chem(z_chem)
            kg_payload = self.kg_action_encoder(
                z_s, z_chem, drug_idx, bank,
                mode=source_mode, edge_dropout=edge_dropout, branch_all=branch_all,
                branch_mask_override=branch_mask_override,
            )
            z_kg = kg_payload["z_kg"]
        else:
            z_chem = self._drop_chem(self.drug_encoder(drug_fp))
            z_kg = torch.zeros_like(z_chem)

        # --- state-adaptive knowledge injection, eq. (12)-(15) ----------------
        if kg_mode in {
            "reviewer_fixed_no_gate",
            "reviewer_average_no_gate",
            "reviewer_concat_no_gate",
        }:
            # The static fusion controls remove the downstream
            # injection gate altogether.  z_kg is fused directly below, so the
            # effective multiplier is exactly one and no gate parameter gets a
            # gradient.  The three modes differ only in how source embeddings
            # are combined: global weights, uniform averaging, or concatenation
            # followed by a static projection.
            gate = torch.ones_like(z_kg)
        elif kg_mode == "strict_fixed_both":
            # One learned gate logit per latent dimension, shared by every
            # tumour-drug pair.  Reuse the Linear bias so parameter count and
            # parameter initialization stay matched to the full model; the
            # context-dependent weight matrix is intentionally bypassed.
            gate = torch.sigmoid(self.gate.bias).unsqueeze(0).expand_as(z_kg)
        elif kg_mode == "strict_uniform_both":
            # Parameter-free, exactly uniform injection strength across all
            # latent dimensions and all tumour-drug pairs.
            gate = torch.full_like(z_kg, 0.5)
        else:
            gate = torch.sigmoid(self.gate(torch.cat([z_s, z_chem, z_kg], dim=1)))
        z_inj = gate * z_kg
        z_a = z_chem + z_inj

        # --- tumour-drug interaction, eq. (16)-(18) ---------------------------
        b = self.terminal(torch.cat([z_s, z_a, z_s * z_a], dim=1))

        raw_auc_hat = self.raw_auc_head(b).squeeze(1)                        # eq. (19)
        value_hat = torch.sigmoid(self.relative_value_head(b)).squeeze(1)    # eq. (20)/(22)
        cell_residual_hat = self.cell_residual_head(torch.cat([b, z_s, z_a], dim=1)).squeeze(1)

        out = {
            "raw_auc_hat": raw_auc_hat,
            "value_hat": value_hat,
            "cell_residual_hat": cell_residual_hat,
            # inference-time fused absolute AUC (fusion_weight selected on validation;
            # 0.0 during training so the two heads stay decoupled)
            "auc_hat": raw_auc_hat + float(fusion_weight) * cell_residual_hat,
            "gate": gate,
            "z_s": z_s,
            "z_chem": z_chem,
            "z_kg": z_kg,
            "z_a": z_a,
            "b": b,
        }
        out.update({k: v for k, v in kg_payload.items() if k != "z_kg"})
        return out


# ---------------------------------------------------------------------------
# 6. Training objectives  (Methods eq. 23-34)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingObjectives:
    """Active configuration of the manuscript run. Objectives present in the
    codebase but weighted 0 (drug-centred, advantage-alias ranking, drug-
    specificity, KG graph-consistency) contribute no gradients."""

    loss_raw_weight: float = 0.5                      # absolute AUC
    loss_value_weight: float = 0.25                   # relative-sensitive-value
    loss_cell_residual_weight: float = 2.0            # tumour-normalised residual
    loss_within_drug_rank_weight: float = 0.35        # within-drug ranking
    loss_same_cell_cross_drug_rank_weight: float = 0.5  # same-cell cross-drug ranking
    rank_margin: float = 0.05
    advantage_margin: float = 0.05
    min_relative_value_gap: float = 0.15
    min_auc_gap: float = 0.05


def pairwise_margin_loss(
    scores: torch.Tensor,
    target: torch.Tensor,
    response: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    min_target_gap: float,
    min_response_gap: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped pairwise margin-ranking loss, eq. (28)-(29).

    A pair (i, j) within a group is retained when |y_i - y_j| > min_target_gap
    OR |AUC_i - AUC_j| > min_response_gap. Loss is averaged within each group and
    then across the groups that retained at least one pair.

    Fully vectorised over groups: rows are sorted by group, padded into a dense
    [n_groups, max_group, max_group] pair block and reduced with masks, so the
    whole objective costs a fixed handful of kernels instead of one Python
    iteration (and one host-device sync) per group. Peak memory is therefore
    O(n_groups * max_group^2), which is negligible at the batch sizes used here
    (<=512 rows per batch, <=32 rows per group).
    """
    zero = scores.new_zeros(())
    valid = torch.isfinite(target)
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if idx.numel() < 2:
        return zero, zero
    s, y, r, g = (t.index_select(0, idx) for t in (scores, target, response, group_ids))
    g, order = torch.sort(g, stable=True)
    s, y, r = (t.index_select(0, order) for t in (s, y, r))

    counts = torch.unique_consecutive(g, return_counts=True)[1]
    max_n = int(counts.max())                      # only host-device sync
    if max_n < 2:
        return zero, zero
    starts = torch.cumsum(counts, 0) - counts
    ar = torch.arange(max_n, device=s.device)
    row_valid = ar.unsqueeze(0) < counts.unsqueeze(1)                 # [G, max_n]
    pos = (starts.unsqueeze(1) + ar.unsqueeze(0)) * row_valid         # [G, max_n]

    def pad(t: torch.Tensor) -> torch.Tensor:
        return t.index_select(0, pos.reshape(-1)).reshape(pos.shape)

    Y, R, S = pad(y), pad(r), pad(s)
    dY = Y.unsqueeze(2) - Y.unsqueeze(1)           # dY[g, i, j] = Y[g, i] - Y[g, j]
    dR = R.unsqueeze(2) - R.unsqueeze(1)
    dS = S.unsqueeze(2) - S.unsqueeze(1)
    upper = torch.triu(torch.ones((max_n, max_n), dtype=torch.bool, device=s.device), diagonal=1)
    keep = (
        upper.unsqueeze(0)
        & row_valid.unsqueeze(2)
        & row_valid.unsqueeze(1)
        & ((dY.abs() > min_target_gap) | (dR.abs() > min_response_gap))
        & (dY != 0)
    )
    kf = keep.to(dS.dtype)
    pair_loss = F.relu(margin - torch.sign(dY) * dS) * kf
    n_pairs_per_group = kf.sum(dim=(1, 2))
    group_loss = pair_loss.sum(dim=(1, 2)) / n_pairs_per_group.clamp_min(1.0)
    n_groups = (n_pairs_per_group > 0).sum()
    return group_loss.sum() / n_groups.clamp_min(1), n_pairs_per_group.sum()


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 over finite targets only (eq. 25-27)."""
    m = torch.isfinite(target)
    if not bool(m.any()):
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[m], target[m])


def gauge_loss(
    out: dict[str, torch.Tensor],
    *,
    auc: torch.Tensor,
    value_train: torch.Tensor,
    cell_residual_train: torch.Tensor,
    drug_group_id: torch.Tensor,
    cell_group_id: torch.Tensor,
    cfg: TrainingObjectives = TrainingObjectives(),
    ranking_enabled: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total active loss, eq. (24):

        L = 0.50 * L_absolute_AUC          (Smooth L1 on raw AUC)
          + 0.25 * L_RTV                   (Smooth L1 on within-drug percentile)
          + 2.00 * L_cell_residual         (Smooth L1 on tumour-normalised residual)
          + 0.35 * L_within_drug_rank      (pairwise margin, groups = drugs)
          + 0.50 * L_same_cell_cross_drug_rank (pairwise margin, groups = cell lines)
    """
    zero = out["raw_auc_hat"].new_zeros(())
    l_auc = F.smooth_l1_loss(out["raw_auc_hat"], auc)
    l_value = _masked_smooth_l1(out["value_hat"], value_train)
    l_resid = _masked_smooth_l1(out["cell_residual_hat"], cell_residual_train)

    l_rank_drug, l_rank_cell = zero, zero
    if ranking_enabled:
        # within-drug: order cell lines for the same drug by RTV
        l_rank_drug, _ = pairwise_margin_loss(
            out["value_hat"], value_train, auc, drug_group_id,
            min_target_gap=cfg.min_relative_value_gap,
            min_response_gap=cfg.min_auc_gap,
            margin=cfg.rank_margin,
        )
        # same-cell cross-drug: order drugs within one cell-line background by residual
        l_rank_cell, _ = pairwise_margin_loss(
            out["cell_residual_hat"], cell_residual_train, auc, cell_group_id,
            min_target_gap=cfg.min_relative_value_gap,
            min_response_gap=cfg.min_auc_gap,
            margin=cfg.advantage_margin,
        )

    loss = (
        cfg.loss_raw_weight * l_auc
        + cfg.loss_value_weight * l_value
        + cfg.loss_cell_residual_weight * l_resid
        + cfg.loss_within_drug_rank_weight * l_rank_drug
        + cfg.loss_same_cell_cross_drug_rank_weight * l_rank_cell
    )
    # Returned as detached tensors, not floats: converting here would force a
    # host-device sync on every optimiser step. Callers accumulate the tensors
    # and call .item() once per epoch.
    parts = {
        "loss_total": loss.detach(),
        "loss_absolute_auc": l_auc.detach(),
        "loss_value": l_value.detach(),
        "loss_cell_residual": l_resid.detach(),
        "loss_within_drug_rank": l_rank_drug.detach(),
        "loss_same_cell_cross_drug_rank": l_rank_cell.detach(),
    }
    return loss, parts


# ---------------------------------------------------------------------------
# 7. Minimal training / inference driver
# ---------------------------------------------------------------------------


def train_epoch(
    model: GAUGE,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[dict[str, torch.Tensor]],
    *,
    cfg: TrainingObjectives = TrainingObjectives(),
    edge_dropout: float = 0.1,
) -> dict[str, float]:
    """One epoch. Each batch dict carries: state, drug_idx, auc, value_train,
    cell_residual_train, drug_group_id, cell_group_id, and optionally
    `is_rank_batch` (hybrid drug-level sampler: ranking losses are applied only
    to the grouped-rank fraction of batches)."""
    model.train()
    sums: dict[str, torch.Tensor] = {}
    n = 0
    for batch in batches:
        out = model(
            batch["state"],
            drug_idx=batch["drug_idx"],
            edge_dropout=edge_dropout,
            fusion_weight=0.0,
        )
        loss, parts = gauge_loss(
            out,
            auc=batch["auc"],
            value_train=batch["value_train"],
            cell_residual_train=batch["cell_residual_train"],
            drug_group_id=batch["drug_group_id"],
            cell_group_id=batch["cell_group_id"],
            cfg=cfg,
            ranking_enabled=bool(batch.get("is_rank_batch", True)),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        bn = int(batch["auc"].shape[0])
        n += bn
        for k, v in parts.items():
            sums[k] = sums[k] + v * bn if k in sums else v * bn
    return {k: float(v) / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def predict(
    model: GAUGE,
    state: torch.Tensor,
    drug_idx: torch.Tensor,
    *,
    kg_mode: str = "multikg_gat",
    fusion_weight: float = 0.0,
    branch_all: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    model.eval()
    if branch_all is None:
        branch_all = model.precompute_branch_all(edge_dropout=0.0)
    return model(
        state,
        drug_idx=drug_idx,
        kg_mode=kg_mode,
        edge_dropout=0.0,
        branch_all=branch_all,
        fusion_weight=fusion_weight,
    )


def build_optimizer(model: GAUGE, lr: float = 3e-6, weight_decay: float = 1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def architecture_summary(model: GAUGE, state_dim: int) -> dict[str, object]:
    return {
        "model": "GAUGE (state-adaptive knowledge-graph gating)",
        "state_encoder": f"{state_dim}->256->128",
        "drug_encoder": "MorganFP2048->512->128",
        "kg_branches": list(getattr(model.kg_action_encoder, "branch_names", [])),
        "kg_gat": "2 x relation-aware graph attention layers per source",
        "kg_source_attention": f"{LATENT_DIM * 3 + 2}->128->1 (context-conditioned, coverage-masked softmax)",
        "gating": "g = sigmoid(W[z_s, z_chem, z_kg]); z_a = z_chem + g * z_kg",
        "interaction": "b = T([z_s, z_a, z_s*z_a]) : 384->256->128",
        "heads": ["raw_auc_head", "relative_value_head", "cell_residual_head"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
