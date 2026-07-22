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
    DecisionCycleRecord,
    DecisionProposal,
    DecisionStepSignature,
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


class PromptCondition(str, Enum):
    AUTHENTIC = "authentic"
    RECEIPT_ONLY = "receipt_only"
    RENAMED = "renamed"
    SHUFFLED_TOPOLOGY = "shuffled_topology"


class CompletionBackend(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def complete(self, role: str, system: str, payload: Mapping[str, Any]) -> str: ...


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
                    raise
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

    def initialize_binding_from_example(
        self,
        motif: MotifCandidate,
        adaptation_example: Mapping[str, Any],
    ) -> BindingHypothesis | None:
        """One target example initializes a provisional binding; it never admits truth."""
        payload = {
            "source_motif": self._binding_motif_view(motif),
            "one_target_adaptation_example": adaptation_example,
            "registered_verifier_ids": sorted(self.allowed_verifier_ids),
        }
        system = (
            "Propose one provisional cross-domain binding hypothesis or abstain. The source graph contains "
            "anonymous receipt-grounded control structure, not target semantics. The single target example "
            "may initialize a hypothesis but cannot prove it. Do not use a predefined source-to-target ontology. "
            "Return exact JSON with keys abstain, target_claim, testable_prediction, verifier_id. Select verifier_id "
            "from the registry. Never output a target action."
        )
        raw = _strict_json(self.backend.complete("binding", system, payload))
        self.call_receipts.append({
            "phase": "one_shot_binding",
            "payload_sha256": stable_hash(payload),
            "response_sha256": stable_hash(raw),
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
        })
        if raw.get("abstain") is True:
            return None
        verifier_id = str(raw.get("verifier_id", ""))
        if verifier_id not in self.allowed_verifier_ids:
            return None
        example_hash = stable_hash(adaptation_example)
        body = {
            "motif_id": motif.motif_id,
            "target_claim": str(raw.get("target_claim", "")),
            "testable_prediction": str(raw.get("testable_prediction", "")),
            "adaptation_receipts": [example_hash],
            "verifier_id": verifier_id,
        }
        return BindingHypothesis(
            stable_hash(body),
            motif.motif_id,
            body["target_claim"],
            body["testable_prediction"],
            (example_hash,),
            verifier_id,
        )

    def review(self, proposal, observation, binding, history):
        payload = {
            "proposal_id": proposal.proposal_id,
            "already_selected_target_native_action": proposal.action,
            "proposal_prediction": proposal.prediction,
            "observation": observation.state,
            "native_actions": observation.native_actions,
            "binding": asdict(binding) if binding else None,
            "recent_live_transition_receipts": [asdict(row) for row in history[-6:]],
        }
        system = (
            "The Decision Agent has already selected the displayed target-native action. You may inspect it but "
            "must not select, replace, rewrite, rank, or output an action. Treat the provisional binding as an "
            "untrusted hypothesis initialized by one example. Use live receipts to decide whether its advisory "
            "structure is still testable. "
            "Return exact JSON with verdict (ADMIT, REPLAN, or ABSTAIN), reason, current_role, "
            "open_hypotheses, information_need, expected_transition, failure_route, termination_test. "
            "ADMIT allows the already selected action; REPLAN asks the Decision Agent to reconsider; ABSTAIN "
            "disables source intervention. Do not output any action field."
        )
        raw = _strict_json(self.backend.complete("review", system, payload))
        advisory = Advisory(
            AdvisoryVerdict(str(raw["verdict"])),
            str(raw.get("reason", "")),
            tuple(str(value) for value in raw.get("evidence_receipt_ids", [])),
            str(raw.get("current_role", "")),
            tuple(str(value) for value in raw.get("open_hypotheses", [])),
            str(raw.get("information_need", "")),
            str(raw.get("expected_transition", "")),
            str(raw.get("failure_route", "")),
            str(raw.get("termination_test", "")),
        )
        self.call_receipts.append({
            "phase": "review",
            "payload_sha256": stable_hash(payload),
            "response_sha256": stable_hash(raw),
            "usage": dict(getattr(self.backend, "last_usage", {}) or {}),
            "advisory": asdict(advisory),
        })
        return advisory
