from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import hashlib
import http.client
import json
import os
from typing import Any, Mapping, Protocol, Sequence
from urllib import request
from urllib import error as urllib_error

from .contracts import (
    Advisory,
    AdvisoryVerdict,
    BindingHypothesis,
    BindingEvidence,
    DecisionCycleRecord,
    DecisionProposal,
    DecisionStepSignature,
    EvidenceVerdict,
    Lifecycle,
    MotifCandidate,
    MotifEdge,
    MotifNode,
    Observation,
    ReplayForkReceipt,
    SourcePolicyStepRecord,
    SourceStepSignature,
    TransitionReceipt,
    stable_hash,
)
from .binding import (
    AttributedBinding,
    BindingArtifactStatus,
    BindingAttribution,
    FrozenBindingArtifact,
    alpha_rename_target_actions,
    validate_structural_binding,
)


class PromptCondition(str, Enum):
    AUTHENTIC = "authentic"
    RECEIPT_ONLY = "receipt_only"
    RENAMED = "renamed"
    SHUFFLED_TOPOLOGY = "shuffled_topology"


class CompletionBackend(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str: ...


class MemoizedCompletionBackend:
    """Common-randomness control: identical requests receive identical completions."""

    def __init__(self, backend: CompletionBackend) -> None:
        self.backend = backend
        self._cache: dict[str, tuple[str, Mapping[str, Any]]] = {}
        self.last_completion = ""
        self.last_usage: Mapping[str, Any] = {}

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"memoized_exact_request": True, "wrapped": self.backend.identity}

    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str:
        key = stable_hash({"role": role, "system": system, "payload": payload})
        if key in self._cache:
            completion, original_usage = self._cache[key]
            self.last_completion = completion
            self.last_usage = {"cache_hit": True, "original_usage": dict(original_usage)}
            return completion
        completion = self.backend.complete(role, system, payload)
        usage = dict(getattr(self.backend, "last_usage", {}) or {})
        self._cache[key] = (completion, usage)
        self.last_completion = completion
        self.last_usage = {"cache_hit": False, "original_usage": usage}
        return completion


class OpenAICompatibleBackend:
    """Frozen inference client. Role-to-model routing can target LoRA-served names."""

    def __init__(
        self,
        base_url: str,
        role_models: Mapping[str, str],
        *,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: int = 180,
        json_mode: bool = False,
        temperature: float | None = 0,
        request_overrides: Mapping[str, Any] | None = None,
        transport_attempts: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.role_models = dict(role_models)
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.json_mode = json_mode
        self.temperature = temperature
        self.request_overrides = dict(request_overrides or {})
        self.transport_attempts = transport_attempts
        self.last_completion = ""
        self.last_usage: Mapping[str, Any] = {}

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "backend": "openai-compatible",
            "base_url": self.base_url,
            "models": self.role_models,
            "temperature": self.temperature,
            "request_overrides": self.request_overrides,
            "transport_attempts": self.transport_attempts,
        }

    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str:
        if role not in self.role_models:
            raise KeyError(f"no frozen model configured for role {role}")
        request_body: dict[str, Any] = {
                "model": self.role_models[role],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
        }
        if self.temperature is not None:
            request_body["temperature"] = self.temperature
        if self.json_mode:
            request_body["response_format"] = {"type": "json_object"}
        request_body.update(self.request_overrides)
        body = json.dumps(request_body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST")
        last_error = None
        for attempt in range(self.transport_attempts):
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read())
                break
            except urllib_error.HTTPError as exc:
                last_error = exc
                if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == self.transport_attempts:
                    detail = exc.read().decode("utf-8", errors="replace")[:2000]
                    raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            except (urllib_error.URLError, http.client.IncompleteRead, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 == self.transport_attempts:
                    raise
        else:  # pragma: no cover - loop always raises or breaks
            raise RuntimeError(f"transport attempts exhausted: {last_error}")
        self.last_completion = str(result["choices"][0]["message"]["content"])
        self.last_usage = result.get("usage") or {}
        return self.last_completion


class TransformersPeftBackend:
    """Single-process base/PEFT backend for a matched frozen comparison.

    Heavy dependencies are imported lazily so the core Harness remains usable
    without a GPU stack.  ``use_adapter_by_role=False`` runs the exact same
    loaded model with LoRA layers disabled rather than changing serving backend.
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: str | None,
        *,
        use_adapter_by_role: Mapping[str, bool],
        max_new_tokens: int = 768,
    ) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:  # pragma: no cover - older transformers
            AutoModelForImageTextToText = None

        self.base_model = base_model
        self.adapter_path = adapter_path
        self.use_adapter_by_role = dict(use_adapter_by_role)
        self.max_new_tokens = max_new_tokens
        self.last_completion = ""
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        is_multimodal = hasattr(config, "text_config") or hasattr(config, "vision_config")
        adapter_namespace = self._adapter_namespace(adapter_path) if adapter_path else None
        if adapter_namespace == "text-only":
            loader = AutoModelForCausalLM
        else:
            loader = (
                AutoModelForImageTextToText
                if is_multimodal and AutoModelForImageTextToText is not None
                else AutoModelForCausalLM
            )
        self.loader_name = loader.__name__
        self.model = loader.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.adapter_name = "segment"
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, adapter_name=self.adapter_name,
            )
        self.model.eval()

    @staticmethod
    def _adapter_namespace(adapter_path: str) -> str:
        from safetensors import safe_open

        weights = os.path.join(adapter_path, "adapter_model.safetensors")
        with safe_open(weights, framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
        text_only = any(".model.model.layers." in key for key in keys)
        umbrella = any(".model.model.language_model.layers." in key for key in keys)
        if text_only == umbrella:
            raise RuntimeError("adapter namespace is mixed or unrecognized")
        return "text-only" if text_only else "multimodal-umbrella"

    @property
    def identity(self) -> Mapping[str, Any]:
        adapter_sha256 = None
        if self.adapter_path:
            path = os.path.join(self.adapter_path, "adapter_model.safetensors")
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            adapter_sha256 = digest.hexdigest()
        return {
            "backend": "transformers-peft",
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "adapter_sha256": adapter_sha256,
            "loader": self.loader_name,
            "use_adapter_by_role": self.use_adapter_by_role,
        }

    def _generate(self, role: str, messages: Sequence[Mapping[str, str]]) -> str:
        import torch

        self.last_completion = ""
        use_adapter = self.use_adapter_by_role.get(role, False)
        if use_adapter and not self.adapter_path:
            raise RuntimeError(f"role {role} requires a missing adapter")
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        context = (
            self.model.disable_adapter()
            if self.adapter_path and not use_adapter
            else _NullContext()
        )
        if use_adapter:
            self.model.set_adapter(self.adapter_name)
        with context, torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion = generated[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True).strip()
        self.last_completion = text
        if not text:
            raise RuntimeError("model returned an empty completion")
        return text

    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str:
        return self._generate(role, [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])

    def complete_prompt(self, role: str, prompt: str) -> str:
        """Native single-user prompt used to train the old segment adapter."""
        return self._generate(role, [{"role": "user", "content": prompt}])


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def _strict_json(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model output must be one JSON object")
    return value


def _bounded_index(values: Sequence[Any], raw_index: Any) -> Any:
    index = int(raw_index)
    if index < 0 or index >= len(values):
        raise ValueError(f"model referenced out-of-range index {index}")
    return values[index]


class FrozenJSONMotifAgent:
    """One conceptual Motif Agent backed by frozen routed model heads."""

    def __init__(
        self,
        backend: CompletionBackend,
        *,
        condition: PromptCondition = PromptCondition.AUTHENTIC,
        allowed_verifier_ids: Sequence[str] = (),
    ) -> None:
        self.backend = backend
        self.condition = condition
        self.allowed_verifier_ids = frozenset(allowed_verifier_ids)
        self.call_receipts: list[Mapping[str, Any]] = []
        self._registered_motifs: dict[str, MotifCandidate] = {}

    def register_motif(self, motif: MotifCandidate) -> None:
        existing = self._registered_motifs.get(motif.motif_id)
        if existing is not None and existing != motif:
            raise ValueError("motif id was registered with different content")
        self._registered_motifs[motif.motif_id] = motif

    def _cycle_view(self, record: DecisionCycleRecord, index: int) -> dict[str, Any]:
        selected = record.proposal_set.selected
        selected_ordinal = next(
            i for i, row in enumerate(record.proposal_set.proposals) if row.proposal_id == selected.proposal_id
        )
        view: dict[str, Any] = {
            "cycle_index": index,
            "cycle_receipt_id": record.receipt.cycle_id,
            "transition_receipt_id": record.transition.receipt_id,
            "proposal_count": len(record.proposal_set.proposals),
            "selected_ordinal": selected_ordinal,
            "post_verdict": record.assessment.verdict.value,
            "continuation_decision": record.assessment.continuation.value,
            "before_hash": record.transition.before_hash,
            "after_hash": record.transition.after_hash,
        }
        if self.condition == PromptCondition.AUTHENTIC:
            view["untrusted_predictions"] = [row.prediction for row in record.proposal_set.proposals]
            view["untrusted_rationales"] = [row.rationale for row in record.proposal_set.proposals]
        return view

    def propose_motifs(self, records, replay_receipts):
        payload = {
            "condition": self.condition.value,
            "cycles": [self._cycle_view(row, index) for index, row in enumerate(records)],
            "replay_forks": [
                {
                    "fork_index": index,
                    "receipt_id": row.receipt_id,
                    "source_transition_id": row.source_transition_id,
                    "fork_state_hash": row.fork_state_hash,
                    "alternative_after_hash": row.alternative_after_hash,
                }
                for index, row in enumerate(replay_receipts)
            ],
        }
        system = (
            "Propose zero or more non-semantic control motifs. Return exact JSON: "
            '{"motifs":[{"nodes":[{"node_id":"n0","cycle_indices":[0]},'
            '{"node_id":"n1","cycle_indices":[1]}],'
            '"edges":[{"source":"n0","target":"n1","fork_indices":[0]}],'
            '"description":"untrusted"}]}. Refer only to supplied integer indices. '
            "For every edge, each referenced replay fork's source_transition_id must equal a "
            "transition_receipt_id represented by that edge's source node, and fork_indices must "
            "include ALL supplied forks from those source-node transitions. Return no motif when "
            "that equality cannot be satisfied. Do not map source actions to target semantics "
            "and do not claim success."
        )
        response = _strict_json(self.backend.complete("segment", system, payload))
        candidates: list[MotifCandidate] = []
        for raw in response.get("motifs", []):
            nodes: list[MotifNode] = []
            for node in raw.get("nodes", []):
                indices = tuple(int(index) for index in node.get("cycle_indices", []))
                selected_records = tuple(_bounded_index(records, index) for index in indices)
                signatures = tuple(
                    DecisionStepSignature(
                        len(record.proposal_set.proposals),
                        next(
                            i
                            for i, proposal in enumerate(record.proposal_set.proposals)
                            if proposal.proposal_id == record.proposal_set.selected_proposal_id
                        ),
                        record.assessment.verdict.value,
                        record.assessment.continuation.value,
                    )
                    for record in selected_records
                )
                nodes.append(
                    MotifNode(
                        str(node["node_id"]),
                        tuple(record.transition.receipt_id for record in selected_records),
                        signatures,
                    )
                )
            edges = tuple(
                MotifEdge(
                    str(edge["source"]),
                    str(edge["target"]),
                    tuple(
                        _bounded_index(replay_receipts, index).receipt_id
                        for index in edge.get("fork_indices", [])
                    ),
                    str(edge.get("claim", "")),
                )
                for edge in raw.get("edges", [])
            )
            content = {
                "lineage": [record.receipt.cycle_id for record in records],
                "nodes": [asdict(node) for node in nodes],
                "edges": [asdict(edge) for edge in edges],
            }
            candidates.append(
                MotifCandidate(
                    stable_hash(content),
                    tuple(content["lineage"]),
                    tuple(nodes),
                    edges,
                    Lifecycle.CANDIDATE,
                    str(raw.get("description", "")),
                )
            )
        return tuple(candidates)

    def _source_step_view(
        self,
        record: SourcePolicyStepRecord,
        index: int,
        *,
        skill_alias: str | None = None,
    ) -> dict[str, Any]:
        action_ordinal = record.before.native_actions.index(record.action)
        view: dict[str, Any] = {
            "step_index": index,
            "transition_receipt_id": record.transition.receipt_id,
            "before_hash": record.transition.before_hash,
            "after_hash": record.transition.after_hash,
            "native_action_count": len(record.before.native_actions),
            "executed_action_ordinal": action_ordinal,
            "skill_conditioned": record.selected_skill_hash is not None,
            "action_origin": record.action_origin,
            "policy_adapter": record.policy_adapter,
            "reward_sign": (
                "POSITIVE" if record.reward > 0
                else "NEGATIVE" if record.reward < 0
                else "ZERO"
            ),
            "terminal": record.after.terminal,
        }
        if self.condition != PromptCondition.RECEIPT_ONLY:
            view["untrusted_source_reasoning"] = record.action_reasoning
            view["untrusted_selected_skill_id"] = (
                skill_alias if skill_alias is not None else record.selected_skill_id
            )
        return view

    def propose_source_motifs(
        self,
        records: Sequence[SourcePolicyStepRecord],
        replay_receipts: Sequence[ReplayForkReceipt],
    ) -> tuple[MotifCandidate, ...]:
        ordered = tuple(sorted(records, key=lambda row: row.step))
        skill_class_by_id: dict[str, int] = {}
        for record in ordered:
            if record.selected_skill_id is not None and record.selected_skill_id not in skill_class_by_id:
                skill_class_by_id[record.selected_skill_id] = len(skill_class_by_id)
        runs: list[tuple[SourcePolicyStepRecord, ...]] = []
        start = 0
        for index in range(1, len(ordered) + 1):
            boundary = (
                index == len(ordered)
                or ordered[index].selected_skill_id != ordered[index - 1].selected_skill_id
                or ordered[index].step != ordered[index - 1].step + 1
            )
            if boundary:
                runs.append(ordered[start:index])
                start = index
        proposal_runs = list(runs)
        if self.condition == PromptCondition.SHUFFLED_TOPOLOGY and len(proposal_runs) > 1:
            proposal_runs.sort(key=lambda run: stable_hash({
                "shuffled_topology_control": [
                    record.transition.receipt_id for record in run
                ]
            }))
            if proposal_runs == runs:
                proposal_runs = proposal_runs[1:] + proposal_runs[:1]
        visible_index = {
            (record.episode_id, record.step): index
            for index, record in enumerate(
                record for run in proposal_runs for record in run
            )
        }
        payload = {
            "condition": self.condition.value,
            "boundary_rule": "MAXIMAL_EXACT_RECORDED_SKILL_ID_RUN_V2",
            "mechanical_skill_runs": [
                {
                    "run_index": run_index,
                    "step_indices": [
                        visible_index[(row.episode_id, row.step)] for row in run
                    ],
                    "steps": [
                        self._source_step_view(
                            row,
                            visible_index[(row.episode_id, row.step)],
                            skill_alias=(
                                f"SKILL_CLASS_{skill_class_by_id[row.selected_skill_id]}"
                                if self.condition == PromptCondition.RENAMED
                                and row.selected_skill_id is not None else None
                            ),
                        )
                        for row in run
                    ],
                }
                for run_index, run in enumerate(proposal_runs)
            ],
            "replay_forks": [
                {
                    "fork_index": index,
                    "receipt_id": row.receipt_id,
                    "source_transition_id": row.source_transition_id,
                    "fork_state_hash": row.fork_state_hash,
                    "alternative_after_hash": row.alternative_after_hash,
                }
                for index, row in enumerate(replay_receipts)
            ],
        }
        system = (
            "Extract zero or more non-semantic source-policy control motifs. "
            "The supplied runs are mechanically fixed maximal runs of the exact recorded "
            "skill ID. You may group whole runs into graph nodes but may not split a run, "
            "move a step, or invent a boundary. "
            "Return exact JSON: "
            '{"motifs":[{"nodes":[{"node_id":"n0","run_indices":[0]},'
            '{"node_id":"n1","run_indices":[1]}],'
            '"edges":[{"source":"n0","target":"n1","fork_indices":[0]}],'
            '"description":"untrusted"}]}. Refer only to supplied integer indices. '
            "The schema is illustrative, not a candidate to copy. Return an empty motifs list "
            "when the supplied receipts do not support at least two distinct nodes. "
            "For every edge, each referenced replay fork's source_transition_id must equal a "
            "transition_receipt_id represented by that edge's source node, and fork_indices must "
            "include ALL supplied forks from those source-node transitions; otherwise return no motif. "
            "Do not invent proposal sets, target predicates, target actions, or success claims."
        )
        response = _strict_json(self.backend.complete("segment", system, payload))
        candidates: list[MotifCandidate] = []
        for raw in response.get("motifs", []):
            nodes: list[MotifNode] = []
            for node in raw.get("nodes", []):
                indices = tuple(int(index) for index in node.get("run_indices", []))
                selected = tuple(
                    record
                    for index in indices
                    for record in _bounded_index(proposal_runs, index)
                )
                signatures = tuple(SourceStepSignature(
                    skill_conditioned=record.selected_skill_hash is not None,
                    action_origin=record.action_origin,
                    reward_sign=(
                        "POSITIVE" if record.reward > 0
                        else "NEGATIVE" if record.reward < 0
                        else "ZERO"
                    ),
                    terminal=record.after.terminal,
                    skill_class_ordinal=(
                        skill_class_by_id.get(record.selected_skill_id)
                        if record.selected_skill_id is not None else None
                    ),
                ) for record in selected)
                nodes.append(MotifNode(
                    str(node["node_id"]),
                    tuple(record.transition.receipt_id for record in selected),
                    signatures,
                ))
            edges = tuple(MotifEdge(
                str(edge["source"]),
                str(edge["target"]),
                tuple(
                    _bounded_index(replay_receipts, index).receipt_id
                    for index in edge.get("fork_indices", [])
                ),
                str(edge.get("claim", "")),
            ) for edge in raw.get("edges", []))
            content = {
                "lineage": [record.transition.receipt_id for record in ordered],
                "nodes": [asdict(node) for node in nodes],
                "edges": [asdict(edge) for edge in edges],
            }
            candidates.append(MotifCandidate(
                stable_hash(content), tuple(content["lineage"]), tuple(nodes), edges,
                Lifecycle.CANDIDATE, str(raw.get("description", "")),
            ))
        return tuple(candidates)

    def initialize_binding(self, motif, adaptation_records):
        payload = {
            "motif_id": motif.motif_id,
            "source_motif": self._binding_motif_view(motif),
            "adaptation_cycles": [self._cycle_view(row, index) for index, row in enumerate(adaptation_records)],
            "registered_verifier_ids": sorted(self.allowed_verifier_ids),
        }
        system = (
            "Propose one provisional target binding or abstain. Return exact JSON with keys "
            "abstain, target_claim, testable_prediction, verifier_id. A verifier_id must be selected "
            "from the supplied registry. Never output a target action."
        )
        raw = _strict_json(self.backend.complete("binding", system, payload))
        self.call_receipts.append({
            "phase": "binding",
            "payload_sha256": stable_hash(payload),
            "response_sha256": stable_hash(raw),
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
        })
        if raw.get("abstain") is True:
            return None
        verifier_id = str(raw.get("verifier_id", ""))
        if verifier_id not in self.allowed_verifier_ids:
            return None
        body = {
            "motif_id": motif.motif_id,
            "target_claim": str(raw.get("target_claim", "")),
            "testable_prediction": str(raw.get("testable_prediction", "")),
            "adaptation_receipts": [row.transition.receipt_id for row in adaptation_records],
            "verifier_id": verifier_id,
        }
        return BindingHypothesis(
            stable_hash(body),
            motif.motif_id,
            body["target_claim"],
            body["testable_prediction"],
            tuple(body["adaptation_receipts"]),
            verifier_id,
        )

    @staticmethod
    def _binding_motif_view(motif: MotifCandidate) -> dict[str, Any]:
        """Expose receipt-grounded anonymous structure, not only graph dimensions."""
        node_ordinals = {node.node_id: index for index, node in enumerate(motif.nodes)}
        return {
            "motif_id": motif.motif_id,
            "status": motif.status.value,
            "source_lineage_sha256": stable_hash(motif.source_lineage),
            "nodes": [
                {
                    "ordinal": index,
                    "receipt_count": len(node.transition_receipt_ids),
                    "decision_signatures": [asdict(signature) for signature in node.decision_signatures],
                }
                for index, node in enumerate(motif.nodes)
            ],
            "edges": [
                {
                    "source_ordinal": node_ordinals[edge.source],
                    "target_ordinal": node_ordinals[edge.target],
                    "replay_receipt_count": len(edge.replay_receipt_ids),
                }
                for edge in motif.edges
            ],
            "untrusted_description": motif.untrusted_description,
        }

    def _propose_binding_set(
        self,
        motif: MotifCandidate,
        adaptation_example: Mapping[str, Any],
        *,
        phase: str,
        max_candidates: int,
    ) -> tuple[BindingHypothesis, ...]:
        payload = {
            "source_motif": self._binding_motif_view(motif),
            "one_target_adaptation_example": adaptation_example,
            "registered_verifier_ids": sorted(self.allowed_verifier_ids),
            "max_candidates": max_candidates,
        }
        system = (
            "Propose zero or more provisional cross-domain structural bindings. The source graph contains "
            "anonymous receipt-grounded control structure, not target semantics. The single target example "
            "may initialize hypotheses but cannot prove them. Do not use a predefined source-to-target ontology. "
            "Every candidate must partition ALL target transitions across every source node exactly once and align "
            "every source edge to one ordered target boundary. Return exact JSON: "
            '{"abstain":false,"bindings":[{"node_alignment":[{"source_node_ordinal":0,'
            '"target_cycle_indices":[0]}],"edge_alignment":[{"source_edge_ordinal":0,'
            '"target_boundary":[0,1]}],"target_claim":"untrusted","testable_prediction":"untrusted",'
            '"verifier_id":"registered-id"}]}. Refer only to supplied ordinals and indices. Never output or select '
            "a target action. Return at most max_candidates candidates."
        )
        raw = _strict_json(self.backend.complete("binding", system, payload))
        self.call_receipts.append({
            "phase": phase,
            "payload_sha256": stable_hash(payload),
            "response_sha256": stable_hash(raw),
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
        })
        if raw.get("abstain") is True:
            return ()
        raw_bindings = raw.get("bindings") or []
        if not isinstance(raw_bindings, list) or len(raw_bindings) > max_candidates:
            raise ValueError("binding Agent returned an invalid candidate count")
        example_hash = stable_hash(adaptation_example)
        candidates = []
        seen_signatures = set()
        for item in raw_bindings:
            verifier_id = str(item.get("verifier_id", ""))
            if verifier_id not in self.allowed_verifier_ids:
                continue
            node_alignment = tuple(sorted(
                (
                    int(row["source_node_ordinal"]),
                    tuple(int(index) for index in row["target_cycle_indices"]),
                )
                for row in item.get("node_alignment", [])
            ))
            edge_alignment = tuple(sorted(
                (
                    int(row["source_edge_ordinal"]),
                    tuple(int(index) for index in row["target_boundary"]),
                )
                for row in item.get("edge_alignment", [])
            ))
            signature = validate_structural_binding(
                motif,
                target_cycle_count=len(adaptation_example.get("transitions", [])),
                node_alignment=node_alignment,
                edge_alignment=edge_alignment,
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            body = {
                "motif_id": motif.motif_id,
                "target_claim": str(item.get("target_claim", "")),
                "testable_prediction": str(item.get("testable_prediction", "")),
                "adaptation_receipts": [example_hash],
                "verifier_id": verifier_id,
                "node_alignment": node_alignment,
                "edge_alignment": edge_alignment,
                "invariance_signature": signature,
            }
            candidates.append(BindingHypothesis(
                binding_id=stable_hash(body),
                motif_id=motif.motif_id,
                target_claim=body["target_claim"],
                testable_prediction=body["testable_prediction"],
                adaptation_receipt_ids=(example_hash,),
                verifier_id=verifier_id,
                node_alignment=node_alignment,
                edge_alignment=edge_alignment,
                invariance_signature=signature,
            ))
        return tuple(candidates)

    def initialize_binding_set_from_example(
        self,
        motif: MotifCandidate,
        adaptation_example: Mapping[str, Any],
        *,
        max_candidates: int = 4,
        require_alpha_invariance: bool = True,
        induction_repetitions: int = 2,
    ) -> tuple[BindingHypothesis, ...]:
        """Compatibility wrapper returning the hypotheses frozen during adaptation."""
        return self.build_binding_artifact(
            motif,
            adaptation_example,
            max_candidates=max_candidates,
            run_alpha_control=require_alpha_invariance,
            induction_repetitions=induction_repetitions,
        ).hypotheses

    def build_binding_artifact(
        self,
        motif: MotifCandidate,
        adaptation_example: Mapping[str, Any],
        *,
        max_candidates: int = 4,
        run_alpha_control: bool = True,
        induction_repetitions: int = 2,
    ) -> FrozenBindingArtifact:
        """Compile Agent proposals using only structural and repeatability checks.

        Repeated raw-example stability is the admission gate.  Full action alpha
        renaming is an attribution control: surviving it supports a content-free
        structural interpretation; failing it does not invalidate a stable
        one-shot target grounding.
        """
        if induction_repetitions < 1:
            raise ValueError("induction_repetitions must be positive")
        self.register_motif(motif)
        renamed_example = alpha_rename_target_actions(adaptation_example)
        first_original: tuple[BindingHypothesis, ...] = ()
        raw_signature_sets: list[set[str]] = []
        alpha_signature_sets: list[set[str]] = []
        repetition_counts = []
        call_start = len(self.call_receipts)
        for repetition in range(induction_repetitions):
            original = self._propose_binding_set(
                motif, adaptation_example,
                phase=f"one_shot_binding_original_r{repetition}",
                max_candidates=max_candidates,
            )
            if repetition == 0:
                first_original = original
            raw_signatures = {row.invariance_signature for row in original}
            raw_signature_sets.append(raw_signatures)
            renamed_count = None
            alpha_signatures: set[str] = set()
            if run_alpha_control:
                renamed = self._propose_binding_set(
                    motif, renamed_example,
                    phase=f"one_shot_binding_alpha_renamed_r{repetition}",
                    max_candidates=max_candidates,
                )
                alpha_signatures = {row.invariance_signature for row in renamed}
                renamed_count = len(renamed)
                alpha_signature_sets.append(alpha_signatures)
            repetition_counts.append({
                "repetition": repetition,
                "original_candidates": len(original),
                "renamed_candidates": renamed_count,
                "within_repetition_alpha_overlap": len(raw_signatures & alpha_signatures),
            })
        stable_raw = set.intersection(*raw_signature_sets) if raw_signature_sets else set()
        stable_alpha = (
            set.intersection(*alpha_signature_sets)
            if run_alpha_control and alpha_signature_sets else set()
        )
        attributed = tuple(
            AttributedBinding(
                row,
                BindingAttribution.GENERIC_STRUCTURAL
                if row.invariance_signature in stable_alpha
                else BindingAttribution.TARGET_GROUNDED_PROVISIONAL,
            )
            for row in first_original
            if row.invariance_signature in stable_raw
        )
        self.call_receipts.append({
            "phase": "one_shot_binding_stability_gate",
            "admission_gate": "REPEATED_RAW_STRUCTURAL_STABILITY",
            "alpha_role": "ATTRIBUTION_CONTROL_ONLY" if run_alpha_control else "NOT_RUN",
            "induction_repetitions": induction_repetitions,
            "repetition_counts": repetition_counts,
            "admitted_candidates": len(attributed),
            "stable_raw_signatures": sorted(stable_raw),
            "stable_alpha_signatures": sorted(stable_alpha),
        })
        receipt_hashes = tuple(
            stable_hash(row) for row in self.call_receipts[call_start:]
        )
        unsigned = {
            "schema_version": 1,
            "motif_id": motif.motif_id,
            "adaptation_example_sha256": stable_hash(adaptation_example),
            "induction_repetitions": induction_repetitions,
            "raw_signature_sets": tuple(tuple(sorted(rows)) for rows in raw_signature_sets),
            "alpha_signature_sets": tuple(tuple(sorted(rows)) for rows in alpha_signature_sets),
            "bindings": [
                {
                    "hypothesis": FrozenBindingArtifact._hypothesis_dict(row.hypothesis),
                    "attribution": row.attribution.value,
                }
                for row in attributed
            ],
            "status": (
                BindingArtifactStatus.ADMITTED.value
                if attributed else BindingArtifactStatus.REJECTED_UNSTABLE.value
            ),
            "backend_identity_sha256": stable_hash(self.backend.identity),
            "call_receipt_hashes": receipt_hashes,
        }
        return FrozenBindingArtifact(
            schema_version=1,
            motif_id=motif.motif_id,
            adaptation_example_sha256=stable_hash(adaptation_example),
            induction_repetitions=induction_repetitions,
            raw_signature_sets=tuple(tuple(sorted(rows)) for rows in raw_signature_sets),
            alpha_signature_sets=tuple(tuple(sorted(rows)) for rows in alpha_signature_sets),
            bindings=attributed,
            status=(
                BindingArtifactStatus.ADMITTED
                if attributed else BindingArtifactStatus.REJECTED_UNSTABLE
            ),
            backend_identity_sha256=stable_hash(self.backend.identity),
            call_receipt_hashes=receipt_hashes,
            artifact_hash=stable_hash(unsigned),
        )

    def initialize_binding_from_example(
        self,
        motif: MotifCandidate,
        adaptation_example: Mapping[str, Any],
    ) -> BindingHypothesis | None:
        candidates = self.initialize_binding_set_from_example(motif, adaptation_example)
        return candidates[0] if candidates else None

    def review_bindings(self, proposal, observation, bindings, history):
        bindings = tuple(sorted(bindings, key=lambda row: row.binding_id))
        if not bindings:
            raise ValueError("review requires at least one binding")
        motifs = {}
        for binding in bindings:
            if binding.motif_id not in self._registered_motifs:
                raise ValueError("review requires the exact registered source motif")
            motifs[binding.motif_id] = self._registered_motifs[binding.motif_id]
        payload = {
            "proposal_id": proposal.proposal_id,
            "already_selected_target_native_action": proposal.action,
            "proposal_prediction": proposal.prediction,
            "observation": observation.state,
            "native_actions": observation.native_actions,
            "binding_candidates": [asdict(binding) for binding in bindings],
            "receipt_grounded_source_motifs": [
                self._binding_motif_view(motifs[motif_id]) for motif_id in sorted(motifs)
            ],
            "recent_live_transition_receipts": [asdict(row) for row in history[-6:]],
        }
        system = (
            "The Decision Agent has already selected the displayed target-native action. You may inspect it but "
            "must not select, replace, rewrite, rank, or output an action. Treat every provisional binding as an "
            "untrusted one-shot hypothesis. Return one exact JSON object with one candidate_verdict for every "
            "supplied binding_id; "
            "do not omit, add, or rank candidates. Each candidate_verdict contains binding_id, verdict (ADMIT, "
            "REPLAN, or ABSTAIN), reason, active_source_node_ordinal, and at least one zero-based "
            "cited_source_receipt_ordinal within that node's receipt_count, plus current_role, "
            "open_hypotheses, information_need, expected_transition, failure_route, and termination_test. "
            "The Harness derives the common verdict by unanimity. Do not output any action field."
        )
        raw = _strict_json(self.backend.complete("review", system, payload))
        raw_verdicts = raw.get("candidate_verdicts") or []
        by_id = {str(row.get("binding_id")): row for row in raw_verdicts}
        expected_ids = {row.binding_id for row in bindings}
        if set(by_id) != expected_ids or len(raw_verdicts) != len(bindings):
            raise ValueError("review did not return the exact binding version space")
        resolved_citations = []
        verdicts = []
        for binding in bindings:
            row = by_id[binding.binding_id]
            verdicts.append(AdvisoryVerdict(str(row["verdict"])))
            motif = motifs[binding.motif_id]
            node_ordinal = int(row["active_source_node_ordinal"])
            node = _bounded_index(motif.nodes, node_ordinal)
            receipt_ordinals = tuple(int(value) for value in row.get("cited_source_receipt_ordinals", []))
            if not receipt_ordinals:
                raise ValueError("review did not cite source evidence")
            resolved_citations.append({
                "binding_id": binding.binding_id,
                "source_node_ordinal": node_ordinal,
                "source_receipt_ordinals": receipt_ordinals,
                "source_receipt_ids": tuple(
                    _bounded_index(node.transition_receipt_ids, ordinal)
                    for ordinal in receipt_ordinals
                ),
            })
        unanimous = len(set(verdicts)) == 1
        common_verdict = verdicts[0] if unanimous else AdvisoryVerdict.ABSTAIN
        scalar_fields = (
            "current_role", "information_need", "expected_transition",
            "failure_route", "termination_test",
        )
        common_scalars = {}
        for field in scalar_fields:
            values = {str(by_id[binding.binding_id].get(field, "")) for binding in bindings}
            common_scalars[field] = next(iter(values)) if len(values) == 1 else ""
        hypothesis_sets = [
            {str(value) for value in by_id[binding.binding_id].get("open_hypotheses", [])}
            for binding in bindings
        ]
        common_hypotheses = tuple(sorted(set.intersection(*hypothesis_sets))) if hypothesis_sets else ()
        advisory = Advisory(
            common_verdict,
            "unanimous version-space verdict" if unanimous else "binding version space disagreed; target-only fallback",
            (),
            common_scalars["current_role"],
            common_hypotheses,
            common_scalars["information_need"],
            common_scalars["expected_transition"],
            common_scalars["failure_route"],
            common_scalars["termination_test"],
        )
        self.call_receipts.append({
            "phase": "review_version_space",
            "payload_sha256": stable_hash(payload),
            "response_sha256": stable_hash(raw),
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
            "advisory": asdict(advisory),
            "candidate_verdicts": [row.value for row in verdicts],
            "unanimous": unanimous,
            "common_semantic_fields": common_scalars,
            "common_open_hypotheses": common_hypotheses,
            "resolved_source_citations": resolved_citations,
        })
        return advisory

    def review(self, proposal, observation, binding, history):
        return self.review_bindings(proposal, observation, (binding,), history)

    def verify_bindings(self, bindings, before, proposal, after, transition, history):
        bindings = tuple(sorted(bindings, key=lambda row: row.binding_id))
        if not bindings:
            raise ValueError("verification requires at least one binding")
        motifs = {}
        for binding in bindings:
            if binding.motif_id not in self._registered_motifs:
                raise ValueError("verification requires the exact registered source motif")
            motifs[binding.motif_id] = self._registered_motifs[binding.motif_id]
        payload = {
            "binding_candidates": [asdict(binding) for binding in bindings],
            "receipt_grounded_source_motifs": [
                self._binding_motif_view(motifs[motif_id]) for motif_id in sorted(motifs)
            ],
            "before_observation": before.state,
            "already_executed_target_native_action": proposal.action,
            "proposal_prediction": proposal.prediction,
            "after_observation": after.state,
            "transition_receipt": asdict(transition),
            "recent_receipt_ids": [row.receipt_id for row in history[-6:]],
        }
        system = (
            "Verify one live transition separately against every supplied provisional binding. Return one exact JSON object "
            "with candidate_evidence, containing exactly one object per binding_id with verdict (SUPPORTED, "
            "REFUTED, or INCONCLUSIVE) and reason. Do not omit, add, rank, or output actions. REFUTED means the "
            "binding cannot explain the observed transition, not merely that the task is unfinished."
        )
        raw = _strict_json(self.backend.complete("verify", system, payload))
        raw_evidence = raw.get("candidate_evidence") or []
        by_id = {str(row.get("binding_id")): row for row in raw_evidence}
        expected_ids = {row.binding_id for row in bindings}
        if set(by_id) != expected_ids or len(raw_evidence) != len(bindings):
            raise ValueError("verifier did not return the exact binding version space")
        evidence = tuple(BindingEvidence(
            binding.binding_id, transition.receipt_id, binding.verifier_id,
            EvidenceVerdict(str(by_id[binding.binding_id]["verdict"])),
        ) for binding in bindings)
        self.call_receipts.append({
            "phase": "verify_version_space",
            "payload_sha256": stable_hash(payload),
            "response_sha256": stable_hash(raw),
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
            "binding_evidence": [asdict(row) for row in evidence],
            "untrusted_reasons": {
                binding_id: str(row.get("reason", "")) for binding_id, row in by_id.items()
            },
        })
        return evidence

    def verify_transition(self, binding, before, proposal, after, transition, history):
        return self.verify_bindings(
            (binding,), before, proposal, after, transition, history,
        )[0]
