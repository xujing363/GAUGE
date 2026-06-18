from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .explainability import KGExplanationBundle, normalize_explanation_level


class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DrugEncoder(nn.Module):
    def __init__(self, fp_dim: int = 2048, hidden_dim: int = 512, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim), nn.ReLU())

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return self.net(fp)


class FrozenPriorAdapter(nn.Module):
    def __init__(self, prior_dim: int, latent_dim: int = 128):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(max(prior_dim, 1), latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim))
        self.prior_dim = prior_dim

    def forward(self, prior: torch.Tensor) -> torch.Tensor:
        if self.prior_dim == 0:
            prior = prior.new_zeros((prior.shape[0], 1))
        return self.adapter(prior)


class RelationGraphAttentionLayer(nn.Module):
    def __init__(self, latent_dim: int, n_relations: int, dropout: float = 0.1):
        super().__init__()
        self.msg = nn.Linear(latent_dim, latent_dim, bias=False)
        self.rel = nn.Embedding(max(n_relations, 1), latent_dim)
        self.att = nn.Linear(latent_dim * 2, 1)
        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, training_edge_dropout: float = 0.0) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src = edge_index[0]
        dst = edge_index[1]
        if self.training and training_edge_dropout > 0:
            keep = torch.rand((src.shape[0],), device=h.device) >= float(training_edge_dropout)
            src = src[keep]
            dst = dst[keep]
            edge_type = edge_type[keep]
            if src.numel() == 0:
                return h
        msg = self.msg(h.index_select(0, src)) + self.rel(edge_type.clamp_min(0))
        score = torch.sigmoid(self.att(torch.cat([h.index_select(0, dst), msg], dim=1)))
        msg = self.dropout(msg * score)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
        deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device))
        agg = agg / deg.clamp_min(1.0)
        return self.norm(h + F.relu(agg))


class MultiKGActionEncoder(nn.Module):
    def __init__(self, kg_artifacts: Any, drug_fingerprint_bank: np.ndarray, latent_dim: int = 128, fp_dim: int = 2048):
        super().__init__()
        self.branch_names = list(getattr(kg_artifacts, "branch_names", ["ChEMBL", "DRKG", "PrimeKG"]))
        self.n_branches = len(self.branch_names)
        self.latent_dim = latent_dim
        self.drug_ids = [int(x) for x in getattr(kg_artifacts, "drug_ids", [])]
        self.drug_to_local = {int(k): int(v) for k, v in getattr(kg_artifacts, "drug_to_local", {}).items()}
        node_table = getattr(kg_artifacts, "node_table", pd.DataFrame())
        edge_table = getattr(kg_artifacts, "edge_table", pd.DataFrame())
        coverage = getattr(kg_artifacts, "coverage", pd.DataFrame())
        self.n_nodes = int(len(node_table))
        node_types = node_table.get("node_type", pd.Series(["node"] * self.n_nodes)).astype(str).tolist()
        type_vocab = {name: i for i, name in enumerate(sorted(set(node_types) | {"node"}))}
        node_type_ids = torch.as_tensor([type_vocab.get(x, 0) for x in node_types], dtype=torch.long)
        self.register_buffer("node_type_ids", node_type_ids, persistent=False)
        self.node_id_embedding = nn.Embedding(max(self.n_nodes, 1), latent_dim)
        self.node_type_embedding = nn.Embedding(max(len(type_vocab), 1), latent_dim)
        fp_bank = np.asarray(drug_fingerprint_bank, dtype=np.float32)
        self.register_buffer("drug_fingerprint_bank", torch.as_tensor(fp_bank, dtype=torch.float32), persistent=False)
        branch_node_ids = np.zeros((len(self.drug_ids), self.n_branches), dtype=np.int64)
        branch_mask = np.zeros((len(self.drug_ids), self.n_branches), dtype=np.float32)
        branch_weight = np.zeros((len(self.drug_ids), self.n_branches), dtype=np.float32)
        branch_degree = np.zeros((len(self.drug_ids), self.n_branches), dtype=np.float32)
        cov = coverage.set_index("DRUG_ID") if not coverage.empty and "DRUG_ID" in coverage.columns else pd.DataFrame()
        for i, drug_id in enumerate(self.drug_ids):
            for j, branch in enumerate(self.branch_names):
                if not cov.empty and drug_id in cov.index:
                    branch_node_ids[i, j] = int(cov.loc[drug_id].get(f"{branch}_node_id", 0))
                    branch_mask[i, j] = float(cov.loc[drug_id].get(f"has_{branch}", 0.0))
                    branch_weight[i, j] = float(cov.loc[drug_id].get(f"source_weight_{branch}", branch_mask[i, j]))
                    branch_degree[i, j] = float(cov.loc[drug_id].get(f"graph_degree_{branch}", 0.0))
        self.register_buffer("branch_node_ids", torch.as_tensor(branch_node_ids, dtype=torch.long), persistent=False)
        self.register_buffer("branch_mask", torch.as_tensor(branch_mask, dtype=torch.float32), persistent=False)
        self.register_buffer("branch_weight", torch.as_tensor(branch_weight, dtype=torch.float32), persistent=False)
        self.register_buffer("branch_degree", torch.as_tensor(branch_degree, dtype=torch.float32), persistent=False)
        self.chem_to_node = nn.Linear(latent_dim, latent_dim)
        self.branches = nn.ModuleList()
        self.edge_indices: list[str] = []
        self.edge_types: list[str] = []
        self.edge_id_names: list[str] = []
        self.edge_id_to_branch_idx: dict[int, int] = {}
        for branch in self.branch_names:
            sub = edge_table.loc[edge_table.get("source", pd.Series(dtype=str)).astype(str).eq(branch)].copy() if not edge_table.empty else pd.DataFrame()
            if not sub.empty and "edge_id" not in sub.columns:
                sub = sub.reset_index(drop=True)
                sub.insert(0, "edge_id", np.arange(len(sub), dtype=np.int64))
            rels = sorted(sub.get("edge_type", pd.Series(dtype=str)).astype(str).unique().tolist()) if not sub.empty else ["self"]
            rel_to_id = {name: i for i, name in enumerate(rels)}
            if sub.empty:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_type = torch.empty((0,), dtype=torch.long)
                edge_ids = torch.empty((0,), dtype=torch.long)
            else:
                edge_index = torch.as_tensor(sub[["src", "dst"]].astype(int).to_numpy().T, dtype=torch.long)
                edge_type = torch.as_tensor([rel_to_id[x] for x in sub["edge_type"].astype(str)], dtype=torch.long)
                edge_ids = torch.as_tensor(sub["edge_id"].astype(int).to_numpy(copy=True), dtype=torch.long)
            idx_name = f"edge_index_{branch.lower()}"
            type_name = f"edge_type_{branch.lower()}"
            id_name = f"edge_ids_{branch.lower()}"
            self.register_buffer(idx_name, edge_index, persistent=False)
            self.register_buffer(type_name, edge_type, persistent=False)
            self.register_buffer(id_name, edge_ids, persistent=False)
            self.edge_indices.append(idx_name)
            self.edge_types.append(type_name)
            self.edge_id_names.append(id_name)
            self.branches.append(nn.ModuleList([RelationGraphAttentionLayer(latent_dim, len(rels)), RelationGraphAttentionLayer(latent_dim, len(rels))]))
            branch_idx = len(self.branches) - 1
            for edge_id in edge_ids.detach().cpu().tolist():
                self.edge_id_to_branch_idx[int(edge_id)] = branch_idx
        self.alpha = nn.Sequential(nn.Linear(latent_dim * 3 + 2, latent_dim), nn.ReLU(), nn.Linear(latent_dim, 1))
        self.concat_projection = nn.Linear(latent_dim * max(self.n_branches, 1), latent_dim)

    def local_indices(self, drug_ids: list[int], device: torch.device | str | None = None) -> torch.Tensor:
        idx = [self.drug_to_local.get(int(x), 0) for x in drug_ids]
        return torch.as_tensor(idx, dtype=torch.long, device=device)

    def _initial_nodes(self, drug_latent_bank: torch.Tensor, device: torch.device) -> torch.Tensor:
        node_ids = torch.arange(max(self.n_nodes, 1), dtype=torch.long, device=device)
        h = self.node_id_embedding(node_ids) + self.node_type_embedding(self.node_type_ids)
        if self.drug_fingerprint_bank.numel() and self.branch_node_ids.numel():
            drug_init = self.chem_to_node(drug_latent_bank)
            for branch_idx in range(self.n_branches):
                ids = self.branch_node_ids[:, branch_idx]
                h = h.index_copy(0, ids, drug_init)
        return h

    def _branch_edge_keep_mask(self, branch_name: str, edge_ids: torch.Tensor, kg_mask: Any | None) -> torch.Tensor | None:
        if kg_mask is None:
            return None
        if isinstance(kg_mask, str):
            if kg_mask == "all_off":
                return torch.zeros((edge_ids.shape[0],), dtype=torch.bool, device=edge_ids.device)
            if kg_mask == f"{branch_name}_off":
                return torch.zeros((edge_ids.shape[0],), dtype=torch.bool, device=edge_ids.device)
            return torch.ones((edge_ids.shape[0],), dtype=torch.bool, device=edge_ids.device)
        if isinstance(kg_mask, dict):
            keep = torch.ones((edge_ids.shape[0],), dtype=torch.bool, device=edge_ids.device)
            source_off = {str(x) for x in kg_mask.get("source_off", [])}
            if branch_name in source_off:
                keep = torch.zeros((edge_ids.shape[0],), dtype=torch.bool, device=edge_ids.device)
            masked_edge_ids = kg_mask.get("edge_ids", [])
            if masked_edge_ids:
                masked = torch.as_tensor(list(map(int, masked_edge_ids)), dtype=torch.long, device=edge_ids.device)
                keep = keep & ~torch.isin(edge_ids, masked)
            return keep
        raise ValueError(f"Unsupported kg_mask: {kg_mask!r}")

    def _effective_branch_mask(self, drug_idx: torch.Tensor, kg_mask: Any | None) -> torch.Tensor:
        mask = self.branch_weight.index_select(0, drug_idx).clone()
        if kg_mask is None:
            return mask
        if isinstance(kg_mask, str):
            if kg_mask == "all_off":
                return torch.zeros_like(mask)
            if kg_mask.endswith("_off"):
                source_name = kg_mask[:-4]
                if source_name in self.branch_names:
                    mask[:, self.branch_names.index(source_name)] = 0.0
                return mask
            return mask
        if isinstance(kg_mask, dict):
            source_off = {str(x) for x in kg_mask.get("source_off", [])}
            for branch_idx, branch_name in enumerate(self.branch_names):
                if branch_name in source_off:
                    mask[:, branch_idx] = 0.0
            return mask
        raise ValueError(f"Unsupported kg_mask: {kg_mask!r}")

    def _can_reuse_precomputed_branches(self, kg_mask: Any | None) -> bool:
        if kg_mask is None:
            return True
        if isinstance(kg_mask, str):
            return kg_mask == "all_off" or kg_mask.endswith("_off")
        if isinstance(kg_mask, dict):
            masked_edge_ids = kg_mask.get("edge_ids", [])
            extra_keys = {str(key) for key in kg_mask.keys()} - {"source_off", "edge_ids"}
            return not extra_keys and not masked_edge_ids
        return False

    def _masked_branch_indices(self, kg_mask: Any | None) -> set[int]:
        if kg_mask is None:
            return set()
        affected: set[int] = set()
        if isinstance(kg_mask, str):
            if kg_mask == "all_off":
                return set(range(self.n_branches))
            if kg_mask.endswith("_off"):
                source_name = kg_mask[:-4]
                if source_name in self.branch_names:
                    affected.add(self.branch_names.index(source_name))
            return affected
        if isinstance(kg_mask, dict):
            source_off = {str(x) for x in kg_mask.get("source_off", [])}
            for branch_idx, branch_name in enumerate(self.branch_names):
                if branch_name in source_off:
                    affected.add(branch_idx)
            for edge_id in kg_mask.get("edge_ids", []):
                branch_idx = self.edge_id_to_branch_idx.get(int(edge_id))
                if branch_idx is not None:
                    affected.add(branch_idx)
            return affected
        return affected

    def _branch_embeddings_for_indices(
        self,
        drug_latent_bank: torch.Tensor,
        device: torch.device,
        branch_indices: set[int],
        *,
        edge_dropout: float = 0.0,
        kg_mask: Any | None = None,
        return_node_states: bool = False,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
        h0 = self._initial_nodes(drug_latent_bank, device)
        outs: dict[int, torch.Tensor] = {}
        node_states: dict[int, torch.Tensor] | None = {} if return_node_states else None
        for branch_idx in sorted(branch_indices):
            h = h0
            edge_index = getattr(self, self.edge_indices[branch_idx])
            edge_type = getattr(self, self.edge_types[branch_idx])
            edge_ids = getattr(self, self.edge_id_names[branch_idx])
            keep = self._branch_edge_keep_mask(self.branch_names[branch_idx], edge_ids, kg_mask)
            if keep is not None:
                edge_index = edge_index[:, keep]
                edge_type = edge_type[keep]
            for layer in self.branches[branch_idx]:
                h = layer(h, edge_index, edge_type, training_edge_dropout=edge_dropout)
            if node_states is not None:
                node_states[branch_idx] = h
            ids = self.branch_node_ids[:, branch_idx]
            outs[branch_idx] = h.index_select(0, ids)
        return outs, node_states

    def branch_embeddings(
        self,
        drug_latent_bank: torch.Tensor,
        device: torch.device,
        edge_dropout: float = 0.0,
        kg_mask: Any | None = None,
        *,
        return_node_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h0 = self._initial_nodes(drug_latent_bank, device)
        outs = []
        node_states = []
        for branch_idx, layers in enumerate(self.branches):
            h = h0
            edge_index = getattr(self, self.edge_indices[branch_idx])
            edge_type = getattr(self, self.edge_types[branch_idx])
            edge_ids = getattr(self, self.edge_id_names[branch_idx])
            keep = self._branch_edge_keep_mask(self.branch_names[branch_idx], edge_ids, kg_mask)
            if keep is not None:
                edge_index = edge_index[:, keep]
                edge_type = edge_type[keep]
            for layer in layers:
                h = layer(h, edge_index, edge_type, training_edge_dropout=edge_dropout)
            node_states.append(h)
            ids = self.branch_node_ids[:, branch_idx]
            outs.append(h.index_select(0, ids))
        branch_all = torch.stack(outs, dim=1) if outs else h0.new_zeros((len(self.drug_ids), 0, self.latent_dim))
        if not return_node_states:
            return branch_all
        branch_node_states = (
            torch.stack(node_states, dim=0) if node_states else h0.new_zeros((0, max(self.n_nodes, 1), self.latent_dim))
        )
        return branch_all, branch_node_states

    def precompute_branch_payload(
        self,
        drug_latent_bank: torch.Tensor,
        device: torch.device | str,
        *,
        edge_dropout: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        device_obj = torch.device(device)
        branch_all, branch_node_states = self.branch_embeddings(
            drug_latent_bank,
            device_obj,
            edge_dropout=edge_dropout,
            return_node_states=True,
        )
        return {
            "drug_latent_bank": drug_latent_bank,
            "branch_all": branch_all,
            "branch_node_states": branch_node_states,
        }

    def forward(
        self,
        z_s: torch.Tensor,
        z_chem: torch.Tensor,
        drug_idx: torch.Tensor,
        drug_latent_bank: torch.Tensor,
        *,
        mode: str = "multikg_gat",
        edge_dropout: float = 0.0,
        disable_state_attention: bool = False,
        precomputed_branch_payload: dict[str, torch.Tensor] | None = None,
        kg_mask: Any | None = None,
        return_branch_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        branch_node_states = None
        can_reuse_precomputed = (
            precomputed_branch_payload is not None
            and "branch_all" in precomputed_branch_payload
            and self._can_reuse_precomputed_branches(kg_mask)
        )
        if can_reuse_precomputed:
            branch_all = precomputed_branch_payload["branch_all"]
            if return_branch_states:
                branch_node_states = precomputed_branch_payload.get("branch_node_states")
        elif (
            precomputed_branch_payload is not None
            and "branch_all" in precomputed_branch_payload
            and not self.training
            and float(edge_dropout) == 0.0
        ):
            affected_branches = self._masked_branch_indices(kg_mask)
            cached_branch_node_states = precomputed_branch_payload.get("branch_node_states")
            can_partial_reuse = bool(affected_branches) and len(affected_branches) < self.n_branches
            if return_branch_states and cached_branch_node_states is None:
                can_partial_reuse = False
            if can_partial_reuse:
                recomputed_branch_all, recomputed_node_states = self._branch_embeddings_for_indices(
                    drug_latent_bank,
                    z_s.device,
                    affected_branches,
                    edge_dropout=edge_dropout,
                    kg_mask=kg_mask,
                    return_node_states=return_branch_states,
                )
                branch_all = precomputed_branch_payload["branch_all"].clone()
                for branch_idx, branch_value in recomputed_branch_all.items():
                    branch_all[:, branch_idx, :] = branch_value
                if return_branch_states and cached_branch_node_states is not None:
                    branch_node_states = cached_branch_node_states.clone()
                    assert recomputed_node_states is not None
                    for branch_idx, node_state in recomputed_node_states.items():
                        branch_node_states[branch_idx] = node_state
            else:
                branch_payload = self.branch_embeddings(
                    drug_latent_bank,
                    z_s.device,
                    edge_dropout=edge_dropout,
                    kg_mask=kg_mask,
                    return_node_states=return_branch_states,
                )
                if return_branch_states:
                    branch_all, branch_node_states = branch_payload
                else:
                    branch_all = branch_payload
        else:
            branch_payload = self.branch_embeddings(
                drug_latent_bank,
                z_s.device,
                edge_dropout=edge_dropout,
                kg_mask=kg_mask,
                return_node_states=return_branch_states,
            )
            if return_branch_states:
                branch_all, branch_node_states = branch_payload
            else:
                branch_all = branch_payload
        z_branch = branch_all.index_select(0, drug_idx)
        mask = self._effective_branch_mask(drug_idx, kg_mask)
        degree = self.branch_degree.index_select(0, drug_idx)
        mode = str(mode or "multikg_gat")
        if mode in {"chembl_only", "drkg_only", "primekg_only"}:
            keep = {"chembl_only": 0, "drkg_only": 1, "primekg_only": 2}[mode]
            branch_keep = torch.zeros_like(mask)
            if keep < branch_keep.shape[1]:
                branch_keep[:, keep] = 1.0
            mask = mask * branch_keep
        elif mode in {"shuffled_prior", "shuffled_mapping"} and z_branch.shape[0] > 1:
            perm = torch.randperm(z_branch.shape[0], device=z_branch.device)
            z_branch = z_branch.index_select(0, perm)
        elif mode in {"random_graph", "random_prior"}:
            z_branch = torch.randn_like(z_branch)
        if mode == "concat_only":
            flat = (z_branch * mask.unsqueeze(-1)).reshape(z_branch.shape[0], -1)
            z_kg = self.concat_projection(flat)
            alpha = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            payload = {
                "z_kg": z_kg,
                "kg_alpha": alpha,
                "kg_mask": mask,
                "kg_degree": degree,
                "z_branch": z_branch,
                "kg_branch_node_ids": self.branch_node_ids.index_select(0, drug_idx),
            }
            if branch_node_states is not None:
                payload["kg_branch_node_states"] = branch_node_states
            return payload
        z_s_for_alpha = torch.zeros_like(z_s) if disable_state_attention or mode == "no_state_attention" else z_s
        pieces = []
        degree_norm = torch.log1p(degree).unsqueeze(-1)
        for branch_idx in range(self.n_branches):
            pieces.append(
                self.alpha(
                    torch.cat(
                        [
                            z_s_for_alpha,
                            z_chem,
                            z_branch[:, branch_idx, :],
                            mask[:, branch_idx : branch_idx + 1],
                            degree_norm[:, branch_idx, :],
                        ],
                        dim=1,
                    )
                )
            )
        logits = torch.cat(pieces, dim=1) if pieces else z_s.new_zeros((z_s.shape[0], 0))
        logits = logits.masked_fill(mask <= 0, -1e9)
        has_any = mask.sum(dim=1, keepdim=True) > 0
        alpha = torch.softmax(logits, dim=1) if logits.numel() else logits
        alpha = torch.where(has_any, alpha * mask, torch.zeros_like(mask))
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        z_kg = (z_branch * alpha.unsqueeze(-1)).sum(dim=1)
        payload = {
            "z_kg": z_kg,
            "kg_alpha": alpha,
            "kg_mask": mask,
            "kg_degree": degree,
            "z_branch": z_branch,
            "kg_branch_node_ids": self.branch_node_ids.index_select(0, drug_idx),
        }
        if branch_node_states is not None:
            payload["kg_branch_node_states"] = branch_node_states
        return payload


class TerminalWorldModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        prior_dim: int = 0,
        fp_dim: int = 2048,
        latent_dim: int = 128,
        kg_artifacts: Any | None = None,
        drug_fingerprint_bank: np.ndarray | None = None,
    ):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim, latent_dim=latent_dim)
        self.drug_encoder = DrugEncoder(fp_dim=fp_dim, latent_dim=latent_dim)
        self.prior_adapter = FrozenPriorAdapter(prior_dim, latent_dim=latent_dim)
        self.kg_action_encoder = None
        if kg_artifacts is not None and drug_fingerprint_bank is not None:
            self.kg_action_encoder = MultiKGActionEncoder(kg_artifacts, drug_fingerprint_bank, latent_dim=latent_dim, fp_dim=fp_dim)
        self.gate = nn.Linear(latent_dim * 3, latent_dim)
        self.terminal = nn.Sequential(
            nn.Linear(latent_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.raw_auc_head = nn.Linear(latent_dim, 1)
        self.relative_value_head = nn.Linear(latent_dim, 1)
        self.cell_residual_head = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
        self.drug_centered_head = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
        self.uncertainty_head = nn.Linear(latent_dim, 1)
        self.binary_response_head = nn.Linear(latent_dim, 1)

    def local_drug_indices(self, drug_ids: list[int], device: torch.device | str | None = None) -> torch.Tensor | None:
        if self.kg_action_encoder is None:
            return None
        return self.kg_action_encoder.local_indices(drug_ids, device=device)

    def precompute_kg_payload(
        self,
        *,
        device: torch.device | str,
        edge_dropout: float = 0.0,
        return_branch_states: bool = True,
    ) -> dict[str, torch.Tensor] | None:
        if self.kg_action_encoder is None:
            return None
        drug_latent_bank = self.drug_encoder(self.kg_action_encoder.drug_fingerprint_bank)
        if return_branch_states:
            return self.kg_action_encoder.precompute_branch_payload(
                drug_latent_bank,
                device=device,
                edge_dropout=edge_dropout,
            )
        device_obj = torch.device(device)
        branch_all = self.kg_action_encoder.branch_embeddings(
            drug_latent_bank,
            device_obj,
            edge_dropout=edge_dropout,
            return_node_states=False,
        )
        return {
            "drug_latent_bank": drug_latent_bank,
            "branch_all": branch_all,
        }

    def forward(
        self,
        state: torch.Tensor,
        drug_fp: torch.Tensor,
        prior: torch.Tensor | None = None,
        prior_mask: torch.Tensor | None = None,
        use_terminal: bool = True,
        drug_idx: torch.Tensor | None = None,
        use_prior: bool = True,
        kg_mode: str = "multikg_gat",
        edge_dropout: float = 0.0,
        disable_state_attention: bool = False,
        compute_kg_consistency: bool = True,
        precomputed_kg_payload: dict[str, torch.Tensor] | None = None,
        state_latent: torch.Tensor | None = None,
        drug_latent: torch.Tensor | None = None,
        drug_latent_bank: torch.Tensor | None = None,
        fusion_weight: float = 1.0,
        return_explanations: bool = False,
        explanation_level: str = "source",
        kg_mask: Any | None = None,
        return_internal_latents: bool = False,
    ) -> dict[str, torch.Tensor]:
        explanation_level = normalize_explanation_level(explanation_level)
        z_s = state_latent if state_latent is not None else self.state_encoder(state)
        kg_payload: dict[str, torch.Tensor] = {}
        if use_prior and self.kg_action_encoder is not None and drug_idx is not None:
            if drug_latent_bank is None and precomputed_kg_payload is not None:
                drug_latent_bank = precomputed_kg_payload.get("drug_latent_bank")
            if drug_latent_bank is None:
                drug_latent_bank = self.drug_encoder(self.kg_action_encoder.drug_fingerprint_bank)
            z_chem = drug_latent if drug_latent is not None else drug_latent_bank.index_select(0, drug_idx)
            kg_payload = self.kg_action_encoder(
                z_s,
                z_chem,
                drug_idx,
                drug_latent_bank,
                mode=kg_mode,
                edge_dropout=edge_dropout,
                disable_state_attention=disable_state_attention,
                precomputed_branch_payload=precomputed_kg_payload,
                kg_mask=kg_mask,
                return_branch_states=return_explanations,
            )
            z_prior = kg_payload["z_kg"]
        else:
            z_chem = drug_latent if drug_latent is not None else self.drug_encoder(drug_fp)
            if prior is None:
                prior = drug_fp.new_zeros((drug_fp.shape[0], self.prior_adapter.prior_dim))
            z_prior = self.prior_adapter(prior)
            if prior_mask is not None:
                z_prior = z_prior * prior_mask.reshape(-1, 1)
            if not use_prior:
                z_prior = torch.zeros_like(z_prior)
        gate = torch.sigmoid(self.gate(torch.cat([z_s, z_chem, z_prior], dim=1)))
        z_a = z_chem + gate * z_prior
        if use_terminal:
            b = self.terminal(torch.cat([z_s, z_a, z_s * z_a], dim=1))
        else:
            b = z_s + z_a
        raw_auc_base = self.raw_auc_head(b).squeeze(1)
        value_hat = torch.sigmoid(self.relative_value_head(b)).squeeze(1)
        cell_residual_hat = self.cell_residual_head(torch.cat([b, z_s, z_a], dim=1)).squeeze(1)
        drug_centered_hat = cell_residual_hat
        auc_hat = raw_auc_base + float(fusion_weight) * cell_residual_hat
        uncertainty = F.softplus(self.uncertainty_head(b)).squeeze(1)
        out = {
            "auc_hat": auc_hat,
            "raw_auc_hat": raw_auc_base,
            "raw_auc_base": raw_auc_base,
            "value_hat": value_hat,
            "cell_residual_hat": cell_residual_hat,
            "drug_centered_hat": drug_centered_hat,
            "uncertainty": uncertainty,
            "uncertainty_hat": uncertainty,
            "binary_logit": self.binary_response_head(b).squeeze(1),
            "gate": gate,
            "terminal_latent": b,
        }
        if return_internal_latents:
            out["state_latent"] = z_s
            out["action_latent"] = z_a
            out["prior_latent"] = z_prior
            out["interaction_latent"] = z_s * z_a
        if compute_kg_consistency and "z_branch" in kg_payload and "kg_alpha" in kg_payload:
            active = kg_payload["kg_mask"].sum().clamp_min(1.0)
            centered = (kg_payload["z_branch"] - kg_payload["z_kg"].unsqueeze(1)).abs()
            out["kg_consistency"] = (centered.mean(dim=2) * kg_payload["kg_mask"]).sum() / active
        out.update(kg_payload)
        if return_explanations:
            total_contribution = None
            if "kg_alpha" in kg_payload and "kg_degree" in kg_payload:
                total_contribution = (kg_payload["kg_alpha"] * torch.log1p(kg_payload["kg_degree"])).sum(dim=1)
            explain_query = F.normalize(z_s + z_a + (z_s * z_a), dim=1)
            out["explanations"] = KGExplanationBundle(
                kg_gate=gate,
                kg_total_contribution=total_contribution,
                kg_source_attention=kg_payload.get("kg_alpha"),
                kg_node_ids=kg_payload.get("kg_branch_node_ids"),
                kg_source_names=list(getattr(self.kg_action_encoder, "branch_names", [])) if self.kg_action_encoder is not None else [],
                extra={
                    "explanation_level": explanation_level,
                    "kg_mask": kg_mask,
                    "kg_degree": kg_payload.get("kg_degree"),
                    "kg_explain_query": explain_query,
                    "kg_branch_node_states": kg_payload.get("kg_branch_node_states"),
                },
            )
        return out


def architecture_dict(model: TerminalWorldModel, state_dim: int, prior_dim: int) -> dict[str, object]:
    kg_enabled = model.kg_action_encoder is not None
    return {
        "model": "GDSC_TCGA_one_step_terminal_world_model",
        "state_encoder": f"{state_dim}->256->128",
        "drug_encoder": "MorganFP2048->512->128",
        "kg_prior_encoder": "TxPert-style MultiKG graph action encoder" if kg_enabled else "disabled",
        "legacy_frozen_prior_adapter": f"{prior_dim}->128 compatibility adapter",
        "action_fusion": "z_a = z_chem + sigmoid(W[z_s,z_chem,z_KG]) * z_KG",
        "terminal_consequence_simulator": "b = T(z_s, z_a, z_s*z_a)",
        "heads": [
            "raw_auc_head",
            "relative_value_head",
            "cell_residual_head",
            "drug_centered_head",
            "uncertainty_head",
            "binary_response_head",
        ],
        "no_model_inputs": ["mutation", "CNV", "stage", "age", "cancer_type", "survival", "pathology", "pathway_gene_sets"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
