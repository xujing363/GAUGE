"""A tool-using LLM agent layered on top of gauge_core's prediction API.

The agent never invents a numeric prediction itself -- every GAUGE number in
its answers comes from a real call into `gauge_core.predict` (or the bundled
data tables), dispatched as an OpenAI-style "function calling" tool. This
keeps the LLM in the role of a natural-language interface and explainer over
GAUGE, not a second, unvalidated source of drug-response numbers.

Beyond the core GAUGE tools, the agent can optionally reach:
- **local** knowledge-graph / explainability tools (`gauge_core.kg_tools`), and
- **external** biomedical databases (`gauge_core.bio_tools`: OpenTargets,
  PubChem, UniProt, Europe PMC) when `enable_external=True` -- giving it the
  target-biology context a "virtual disease biologist" needs, with literature
  it can cite.

It runs in two modes:
- `run_turn` -- a planning-aware, multi-round conversational answer, with an
  optional self-reflection pass.
- `run_report` -- a deep-research mode (plan -> gather evidence via tools ->
  synthesise a structured, cited report -> self-critique and revise).

Works with any OpenAI-API-compatible chat endpoint via `gauge_core.providers`,
which also provides multi-provider fallback. Defaults to DeepSeek.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from . import _drugwm_path  # noqa: F401  must precede numpy/pandas/torch (see module docstring)
from ._env import load_dotenv

from . import bio_tools, providers
from .bundle import ModelBundle
from .kg_tools import LOCAL_TOOL_IMPLS, LOCAL_TOOL_SCHEMAS
from .predict import (
    DrugNotFoundError,
    SampleResolutionError,
    predict_one,
    rank_drugs,
    score_combination,
    search_cell_lines,
    search_drugs,
)

load_dotenv()

DEFAULT_MODEL = "deepseek-chat"
# Deep-report synthesis/critique budget. Reasoning models also spend tokens on
# hidden chain-of-thought, so this must comfortably exceed the visible report.
_REPORT_MAX_TOKENS = 6000

SYSTEM_PROMPT = """You are the GAUGE Assistant, a natural-language interface to the GAUGE \
drug-response prediction model, embedded in the GAUGE software for students, clinicians, \
and biologists who may have no programming background. You act like a careful disease \
biologist: you plan, gather evidence with tools, and explain.

How to work:
- First think about what the user is really asking and which tools you need, then call them. \
For multi-part questions, break the problem down and make several tool calls before answering.
- GAUGE drug-response numbers ALWAYS come from the GAUGE tools (predict/rank/combine/explain). \
External-database tools provide biological/pharmacological CONTEXT only -- never treat them as a \
source of a response prediction, and keep the two clearly separate.
- Not every question needs a GAUGE prediction. Pure drug-sensitivity / pharmacology questions \
(\"what drugs target EGFR?\", \"what is erlotinib's mechanism and clinical phase?\", \"how often \
is TP53 mutated?\", \"are there trials of this drug in this cancer?\", \"what pathways is this \
target in?\") can be answered with the external tools alone, without calling GAUGE.

External-database tools you may have (only when external databases are enabled):
- lookup_target_disease_associations (OpenTargets), lookup_protein (UniProt), lookup_pathways \
(Reactome) -- target & disease biology.
- lookup_compound (PubChem), lookup_drug_mechanism (ChEMBL: mechanism, target, clinical phase, \
indications) -- compound pharmacology.
- lookup_drug_gene_interactions (DGIdb) -- which drugs act on a gene/target.
- lookup_cancer_mutations (cBioPortal) -- pan-cancer mutation frequency of a gene.
- search_clinical_trials (ClinicalTrials.gov) and search_literature (Europe PMC) -- clinical \
activity and citable references.

What GAUGE predicts, in the units your tools return:
- `relative_sensitive_value` (0 to 1): the headline, cross-drug-comparable score. Higher \
means GAUGE expects this tumour/cell line to respond better to this drug relative to the \
typical sample tested against it. This is the number to lead with.
- `predicted_absolute_auc`: a raw dose-response-curve AUC estimate. In this convention, a \
LOWER absolute_auc indicates a MORE sensitive (better) response, not a worse one -- do not say \
"high AUC means a strong/confident response". It can also fall slightly outside [0, 1]; \
that is expected model behaviour, not an error.
- `kg_source_attention`: how much each of the three knowledge graphs (ChEMBL = mechanism of \
action, DRKG = gene/protein drug-target edges, PrimeKG = protein-disease edges) contributed.

File uploads: if the user has uploaded an expression file, call list_uploaded_samples to \
see which samples are available from that file, then use those sample names in \
predict_drug_response or rank_drugs_for_sample just like any other sample ID.

Hard rules:
1. NEVER invent or guess a numeric GAUGE prediction yourself. Always call a tool to get real \
numbers, even if you think you already know the answer.
2. If a drug or cell-line name you were given doesn't resolve, call search_drugs or \
search_cell_lines to find the closest real match in the library, and tell the user what \
you found instead of giving up silently.
3. GAUGE is a research tool for hypothesis generation, not a clinical diagnostic or a \
substitute for a treating physician. When a user's question implies a treatment decision, \
say so plainly alongside your answer.
4. Be concise. Cite the actual numbers from tool results rather than only describing them \
qualitatively. When you use literature, cite it by PMID/DOI.
5. Combination scores are a heuristic over two independent single-agent predictions -- GAUGE \
has never seen real combination-response labels. Say so when asked about combinations.
"""

# ── Core GAUGE tool schemas (unchanged from the original agent) ───────────────
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "predict_drug_response",
            "description": (
                "Predict GAUGE's relative sensitive value and absolute AUC for one sample "
                "(a known cell line ID, e.g. SIDM00003) treated with one drug. Use search_cell_lines "
                "or search_drugs first if you are not sure of the exact name/ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_line": {"type": "string", "description": "Known SANGER_MODEL_ID or DepMap ID, e.g. SIDM00003"},
                    "drug": {"type": "string", "description": "Drug name (library) or a raw SMILES string for a novel compound"},
                },
                "required": ["cell_line", "drug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_drugs_for_sample",
            "description": "Rank the bundled drug library from most to least promising for one known sample.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_line": {"type": "string", "description": "Known SANGER_MODEL_ID or DepMap ID"},
                    "top_k": {"type": "integer", "description": "How many top drugs to return (default 10)"},
                },
                "required": ["cell_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_drug_combination",
            "description": (
                "Heuristic combination score for two drugs on one sample, derived from two "
                "independent single-agent GAUGE predictions (not a trained synergy model)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cell_line": {"type": "string"},
                    "drug_a": {"type": "string"},
                    "drug_b": {"type": "string"},
                    "mode": {"type": "string", "enum": ["bliss", "activity_product", "complementarity"]},
                },
                "required": ["cell_line", "drug_a", "drug_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_drugs",
            "description": "Substring search over the bundled drug library to resolve an ambiguous or partial drug name.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cell_lines",
            "description": "Substring search over bundled cell-line metadata (name, tissue, cancer type) to resolve an ambiguous sample.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_uploaded_samples",
            "description": (
                "List the sample names that were uploaded via the expression file uploader. "
                "Call this first when the user asks about their own data or uploaded file."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _tool_predict(bundle: ModelBundle, uploaded: dict, cell_line: str, drug: str) -> dict[str, Any]:
    sample: str | dict = uploaded.get(cell_line, cell_line)
    try:
        result = predict_one(bundle, sample, drug)
    except (DrugNotFoundError, SampleResolutionError) as exc:
        return {"error": str(exc)}
    return {
        "relative_sensitive_value": round(result.value_hat, 3),
        "predicted_absolute_auc": round(result.auc_hat, 3),
        "kg_source_attention": {k: round(v, 3) for k, v in result.kg_alpha.items()} if result.kg_alpha else None,
        "drug_known_in_library": result.drug.known,
        "sample_is_uploaded": cell_line in uploaded,
        "percentile_note": result.percentile_text,
    }


def _tool_rank(bundle: ModelBundle, uploaded: dict, cell_line: str, top_k: int = 10) -> dict[str, Any]:
    sample: str | dict = uploaded.get(cell_line, cell_line)
    try:
        ranked = rank_drugs(bundle, sample)
    except (DrugNotFoundError, SampleResolutionError) as exc:
        return {"error": str(exc)}
    top = ranked.head(int(top_k))[["DRUG_NAME", "value_hat", "auc_hat"]].round(3).rename(
        columns={"value_hat": "relative_sensitive_value", "auc_hat": "absolute_auc"}
    )
    return {"ranked_drugs": top.to_dict(orient="records")}


def _tool_combo(bundle: ModelBundle, uploaded: dict, cell_line: str, drug_a: str, drug_b: str, mode: str = "bliss") -> dict[str, Any]:
    sample: str | dict = uploaded.get(cell_line, cell_line)
    try:
        out = score_combination(bundle, sample, drug_a, drug_b, mode=mode)
    except (DrugNotFoundError, SampleResolutionError, ValueError) as exc:
        return {"error": str(exc)}
    return {
        "drug_a": out["drug_a"],
        "drug_b": out["drug_b"],
        "relative_sensitive_value_a": round(out["value_hat_a"], 3),
        "relative_sensitive_value_b": round(out["value_hat_b"], 3),
        "combination_score": round(out["combination_score"], 3),
        "mode": mode,
    }


def _tool_search_drugs(bundle: ModelBundle, uploaded: dict, query: str) -> dict[str, Any]:
    return {"matches": search_drugs(bundle, query)}


def _tool_search_cells(bundle: ModelBundle, uploaded: dict, query: str) -> dict[str, Any]:
    return {"matches": search_cell_lines(bundle, query)}


def _tool_list_uploaded(bundle: ModelBundle, uploaded: dict) -> dict[str, Any]:
    if not uploaded:
        return {"uploaded_samples": [], "note": "No expression file has been uploaded yet."}
    return {"uploaded_samples": sorted(uploaded.keys()), "count": len(uploaded)}


TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "predict_drug_response": _tool_predict,
    "rank_drugs_for_sample": _tool_rank,
    "score_drug_combination": _tool_combo,
    "search_drugs": _tool_search_drugs,
    "search_cell_lines": _tool_search_cells,
    "list_uploaded_samples": _tool_list_uploaded,
}


def _adapt_external(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """External bio tools take only their own kwargs; wrap them to the
    (bundle, uploaded, **kwargs) calling convention the dispatcher uses."""

    def _wrapped(bundle: ModelBundle, uploaded: dict, **kwargs: Any) -> dict[str, Any]:
        return fn(**kwargs)

    return _wrapped


_EXTERNAL_TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    name: _adapt_external(fn) for name, fn in bio_tools.EXTERNAL_TOOL_IMPLS.items()
}


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentTurnResult:
    reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


class AgentNotConfiguredError(RuntimeError):
    pass


class GaugeAgent:
    def __init__(
        self,
        bundle: ModelBundle,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        uploaded_samples: dict[str, Any] | None = None,
        *,
        provider: str = providers.DEFAULT_PROVIDER,
        fallback_providers: list[str] | None = None,
        report_provider: str = providers.DEFAULT_REPORT_PROVIDER,
        enable_external: bool = False,
        reflect: bool = False,
    ):
        """
        uploaded_samples: mapping from sample name to expression dict/Series
            (same format as gauge_core.expression_io.parse_expression_table output).
            When the agent calls predict/rank/combo with a name that appears in this
            dict, the expression vector is used directly instead of a bundle lookup.

        provider / fallback_providers / report_provider: names from
            gauge_core.providers.PROVIDERS. The chat uses `provider` (falling back
            through `fallback_providers`); the deep report synthesises with
            `report_provider` (a reasoning model) and falls back to `provider`.
        enable_external: expose the external biomedical-database tools (needs internet).
        reflect: run a self-critique/revise pass after a tool-using chat answer.
        """
        self.bundle = bundle
        self.uploaded_samples: dict[str, Any] = uploaded_samples or {}
        self.enable_external = enable_external
        self.reflect = reflect

        self.primary = providers.get_provider(provider)
        self.fallbacks = [providers.get_provider(p) for p in (fallback_providers or [])]
        self.report_provider = providers.get_provider(report_provider)

        # `api_key`/`base_url`/`model` are legacy overrides for the primary provider.
        self._api_keys: dict[str, str] = {}
        if api_key:
            # A pasted key serves the chat provider and any same-vendor provider
            # (e.g. a DeepSeek key also unlocks deepseek-reasoner for report mode).
            for p in (self.primary, self.report_provider, *self.fallbacks):
                if p.env_key == self.primary.env_key:
                    self._api_keys[p.name] = api_key
        if base_url or (model and model != self.primary.default_model):
            # Build a one-off custom provider so legacy callers pointing at a
            # different endpoint/model keep working unchanged.
            self.primary = providers.LLMProvider(
                name=self.primary.name,
                label=self.primary.label,
                base_url=base_url or providers.provider_base_url(self.primary),
                env_key=self.primary.env_key,
                default_model=model or self.primary.default_model,
                supports_tools=self.primary.supports_tools,
                is_reasoning=self.primary.is_reasoning,
            )

        # The assistant relies on function calling, so the chat provider MUST be
        # tool-capable. Reasoning models (no tool support) are only valid for the
        # report's tool-less synthesis step, never as the primary.
        if not self.primary.supports_tools:
            raise AgentNotConfiguredError(
                f"Model {self.primary.name!r} does not support tool calling and cannot be used as the "
                "chat model. Pick a tool-capable model (e.g. deepseek-chat or OpenAI); reasoning models "
                "are used automatically for Deep Report synthesis."
            )
        # Fail fast (as the original agent did) if the primary provider has no key.
        if not providers.resolve_key(self.primary, self._api_keys.get(self.primary.name)):
            raise AgentNotConfiguredError(
                f"No API key configured for {self.primary.name!r}. Set {self.primary.env_key} "
                "in your environment / .env, or enter a key in the sidebar."
            )
        # Surface the missing-openai-package case early, like the original agent.
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise AgentNotConfiguredError("The 'openai' package is required for the GAUGE Assistant.") from exc

    # ── tool registry (depends on enable_external) ────────────────────────────
    def _tool_schemas(self) -> list[dict[str, Any]]:
        schemas = [*TOOL_SCHEMAS, *LOCAL_TOOL_SCHEMAS]
        if self.enable_external:
            schemas += bio_tools.EXTERNAL_TOOL_SCHEMAS
        return schemas

    def _tool_impls(self) -> dict[str, Callable[..., dict[str, Any]]]:
        impls = {**TOOL_IMPLS, **LOCAL_TOOL_IMPLS}
        if self.enable_external:
            impls.update(_EXTERNAL_TOOL_IMPLS)
        return impls

    def _complete(self, primary, messages, *, tools=None, max_tokens=800, **kw):
        call: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, **kw}
        if tools is not None:
            call["tools"] = tools
        return providers.chat_completion(
            primary, fallbacks=[self.primary, *self.fallbacks], api_keys=self._api_keys, **call
        )

    def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        *,
        primary,
        max_tool_rounds: int,
        max_tokens: int = 800,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> tuple[str, list[ToolCallRecord]]:
        """Drive the function-calling loop, returning (final_text, tool_call_log).

        ``progress`` (optional) is called as ``progress("tool", {"record": rec})``
        right after each tool runs, so a UI can show the evidence trail live.
        """
        impls = self._tool_impls()
        schemas = self._tool_schemas()
        tool_call_log: list[ToolCallRecord] = []
        for _ in range(max_tool_rounds):
            resp = self._complete(primary, messages, tools=schemas, max_tokens=max_tokens)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or "", tool_call_log
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                impl = impls.get(name)
                result = impl(self.bundle, self.uploaded_samples, **args) if impl is not None else {"error": f"Unknown tool {name!r}"}
                record = ToolCallRecord(name=name, args=args, result=result)
                tool_call_log.append(record)
                if progress is not None:
                    progress("tool", {"record": record})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})
        return (
            "I made several tool calls but couldn't reach a final answer -- please rephrase or simplify your question.",
            tool_call_log,
        )

    def run_turn(
        self,
        history: list[dict[str, Any]],
        max_tool_rounds: int = 8,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> AgentTurnResult:
        """`history` is a list of {"role": "user"|"assistant", "content": str} dicts
        (no tool messages -- those are internal to this call)."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
        reply, tool_call_log = self._run_tool_loop(
            messages, primary=self.primary, max_tool_rounds=max_tool_rounds, progress=progress
        )
        if self.reflect and tool_call_log:
            reply = self._reflect(messages, reply, tool_call_log)
        return AgentTurnResult(reply=reply, tool_calls=tool_call_log)

    def _reflect(self, messages: list[dict[str, Any]], reply: str, tool_call_log: list[ToolCallRecord]) -> str:
        """One cheap critic+revise pass: catch GAUGE numbers not backed by a tool
        result and a missing research-use caveat, then rewrite if needed."""
        critic_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a strict reviewer of a GAUGE Assistant answer. Check ONLY: "
                    "(1) every numeric GAUGE prediction in the answer matches a value present in the tool "
                    "results; (2) a research-use (not clinical advice) caveat is present if a treatment "
                    "decision is implied. If the answer is already fine, reply with exactly 'OK'. "
                    "Otherwise reply with a corrected version of the answer only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "TOOL RESULTS:\n"
                    + json.dumps([{"name": t.name, "result": t.result} for t in tool_call_log], default=str)
                    + "\n\nANSWER:\n"
                    + reply
                ),
            },
        ]
        try:
            resp = self._complete(self.primary, critic_msgs, max_tokens=800)
            verdict = (resp.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001 - reflection is best-effort
            return reply
        if verdict and verdict.upper() != "OK" and not verdict.upper().startswith("OK"):
            return verdict
        return reply

    # ── deep-research report mode ─────────────────────────────────────────────
    def run_report(
        self,
        history: list[dict[str, Any]],
        max_tool_rounds: int = 12,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> AgentTurnResult:
        """Deep-research mode: plan -> gather evidence with tools -> synthesise a
        structured, cited Markdown report -> self-critique and revise.

        Tool-gathering uses the tool-capable primary provider; synthesis and
        critique use the (stronger, no-tools) reasoning report provider, falling
        back to the primary if the report provider has no key.

        ``progress`` (optional) is called with ("plan"|"gather"|"tool"|"synthesis"
        |"critique"|"done", info) at each stage so a UI can display the live
        research process.
        """
        def _emit(stage: str, info: dict[str, Any] | None = None) -> None:
            if progress is not None:
                progress(stage, info or {})

        question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")

        # 1. Plan -------------------------------------------------------------
        plan_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Draft a brief research plan (3-6 bullet sub-questions and which tools to use) for "
                    f"answering this as a thorough report. Do NOT answer yet.\n\nQUESTION: {question}"
                ),
            },
        ]
        _emit("plan", {"text": "Drafting a research plan…"})
        try:
            plan_resp = self._complete(self.primary, plan_msgs, max_tokens=500)
            plan = plan_resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            plan = ""
        _emit("plan", {"text": plan, "final": True})

        # 2. Gather evidence with tools --------------------------------------
        gather_msgs: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": (
                    "Use your tools to GATHER all evidence needed for a thorough report on the question "
                    "above. Call GAUGE prediction/explain tools for every number, and (if available) "
                    "external target/protein/compound/literature tools for biological context and citations. "
                    "When you have gathered enough, briefly list the evidence you collected."
                    + (f"\n\nResearch plan:\n{plan}" if plan else "")
                ),
            },
        ]
        _emit("gather", {"text": "Gathering evidence with tools…"})
        evidence_summary, tool_call_log = self._run_tool_loop(
            gather_msgs, primary=self.primary, max_tool_rounds=max_tool_rounds, max_tokens=1200,
            progress=progress,
        )

        evidence_blob = json.dumps(
            [{"tool": t.name, "args": t.args, "result": t.result} for t in tool_call_log], default=str, indent=1
        )

        # 3. Synthesise the report (reasoning model, no tools) ----------------
        report_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Write a structured Markdown research report answering:\n\n{question}\n\n"
                    "Use ONLY the gathered evidence below for any number or claim. Sections:\n"
                    "## Summary\n## Target & disease biology\n## GAUGE model predictions\n"
                    "## External evidence & literature\n## Caveats & limitations\n## References\n\n"
                    "Rules: lead with the GAUGE relative_sensitive_value where relevant; clearly separate "
                    "GAUGE predictions from external evidence; cite literature by PMID/DOI in References; "
                    "include the research-use (not clinical advice) caveat. If a number is not in the evidence, "
                    "say it was not computed rather than inventing it.\n\n"
                    f"GATHERED EVIDENCE (JSON):\n{evidence_blob}\n\n"
                    f"Plan that guided gathering:\n{plan}"
                ),
            },
        ]
        # Reasoning models spend tokens on hidden chain-of-thought, so the visible
        # report needs a generous budget or it gets truncated mid-sentence.
        _emit("synthesis", {"text": f"Writing the report from {len(tool_call_log)} pieces of evidence…"})
        report_resp = self._complete(self.report_provider, report_msgs, max_tokens=_REPORT_MAX_TOKENS)
        report = (report_resp.choices[0].message.content or "").strip()

        # If the reasoning model returned no visible text, retry once on the
        # (faster, tool-less) chat provider so the user still gets a report.
        if not report:
            retry = self._complete(self.primary, report_msgs, max_tokens=2400)
            report = (retry.choices[0].message.content or "").strip()
        if not report:
            report = (
                "I gathered the evidence below but could not synthesise a written report "
                "(the model returned no text). The raw evidence is shown in the trail.\n\n"
                + (evidence_summary or "")
            )
            return AgentTurnResult(reply=report, tool_calls=tool_call_log)

        # 4. Critique & revise (reasoning model, no tools) -------------------
        _emit("critique", {"text": "Reviewing the draft for unsupported claims…"})
        report = self._critique_report(question, report, evidence_blob)
        _emit("done", {})
        return AgentTurnResult(reply=report, tool_calls=tool_call_log)

    def _critique_report(self, question: str, report: str, evidence_blob: str) -> str:
        critic_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a strict scientific reviewer. Review the report for: (1) any GAUGE number not "
                    "present in the evidence; (2) external claims not backed by the evidence/literature; "
                    "(3) GAUGE predictions not clearly separated from external evidence; (4) a missing "
                    "research-use caveat. If all good, reply 'OK'. Otherwise return the FULL corrected report."
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nEVIDENCE (JSON):\n{evidence_blob}\n\nREPORT:\n{report}",
            },
        ]
        try:
            resp = self._complete(self.report_provider, critic_msgs, max_tokens=_REPORT_MAX_TOKENS)
            verdict = (resp.choices[0].message.content or "").strip()
            finished = resp.choices[0].finish_reason != "length"
        except Exception:  # noqa: BLE001 - critique is best-effort
            return report
        # Only accept a rewrite that looks like a complete, fuller report -- never
        # replace a good report with a truncated or trivially short critique reply.
        if (
            verdict
            and verdict.upper() not in {"OK", "OK."}
            and not verdict.upper().startswith("OK\n")
            and finished
            and len(verdict) >= 0.8 * len(report)
        ):
            return verdict
        return report
