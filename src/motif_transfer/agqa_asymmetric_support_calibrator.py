"""Induce relation-conditional temporal support without editing bindings."""
from __future__ import annotations
from dataclasses import asdict,dataclass,replace
from itertools import combinations
from typing import Mapping
from .agqa_interval_reliability_calibrator import GAP_THRESHOLDS,SPREAD_THRESHOLDS,binding_geometry
from .agqa_temporal_support_calibrator import MINIMUM_MAX_INTERVAL_SPANS,maximum_interval_span
from .agqa_view_reliability_calibrator import VIEW_KINDS,singleton_view_kind
from .contracts import stable_hash

@dataclass(frozen=True)
class AsymmetricExample:
    split:str;task_id:str;aggregate_authorized:bool;resolved_relation:str|None;singleton_view:str|None;minimum_cross_pair_gap:int;maximum_within_operand_endpoint_spread:int;maximum_interval_span:int;source_correct:bool;target_native_correct:bool
@dataclass(frozen=True)
class AsymmetricRule:
    allowed_singleton_views:tuple[str,...];minimum_cross_pair_gap:int;maximum_within_operand_endpoint_spread:int;before_minimum_interval_span:int;after_minimum_interval_span:int;training_examples:int;training_authorizations:int;training_wins:int;training_losses:int;training_ties:int;selection_authority:str;rule_sha256:str
    @classmethod
    def from_mapping(cls,v:Mapping[str,object]):
        p=dict(v);p["allowed_singleton_views"]=tuple(p["allowed_singleton_views"]);r=cls(**p);r.validate();return r
    def validate(self):
        b=asdict(self);h=b.pop("rule_sha256")
        if stable_hash(b)!=h:raise ValueError("asymmetric rule hash mismatch")
        if self.minimum_cross_pair_gap not in GAP_THRESHOLDS or self.maximum_within_operand_endpoint_spread not in SPREAD_THRESHOLDS:raise ValueError("geometry threshold escaped class")
        if self.before_minimum_interval_span not in MINIMUM_MAX_INTERVAL_SPANS or self.after_minimum_interval_span not in MINIMUM_MAX_INTERVAL_SPANS:raise ValueError("support threshold escaped class")
def _eval(rows,views,gap,spread,before,after):
    q=[x for x in rows if x.aggregate_authorized and (x.singleton_view is None or x.singleton_view in set(views)) and x.minimum_cross_pair_gap>=gap and x.maximum_within_operand_endpoint_spread<=spread and x.maximum_interval_span>=(before if x.resolved_relation=="before" else after)]
    w=sum(x.source_correct and not x.target_native_correct for x in q);l=sum(x.target_native_correct and not x.source_correct for x in q)
    return {"allowed_singleton_views":list(views),"minimum_cross_pair_gap":gap,"maximum_within_operand_endpoint_spread":spread,"before_minimum_interval_span":before,"after_minimum_interval_span":after,"authorizations":len(q),"wins":w,"losses":l,"ties":len(rows)-w-l,"net_gain":w-l,"rule_description_length":len(views)+4}
def induce_asymmetric_rule(examples):
    rows=tuple(examples);c=[]
    for n in range(4):
      for views in combinations(VIEW_KINDS,n):
       for gap in GAP_THRESHOLDS:
        for spread in SPREAD_THRESHOLDS:
         for before in MINIMUM_MAX_INTERVAL_SPANS:
          for after in MINIMUM_MAX_INTERVAL_SPANS:c.append(_eval(rows,views,gap,spread,before,after))
    x=min(c,key=lambda r:(r["losses"],-r["net_gain"],-r["wins"],-r["authorizations"],r["rule_description_length"],r["minimum_cross_pair_gap"],-r["maximum_within_operand_endpoint_spread"],r["before_minimum_interval_span"]+r["after_minimum_interval_span"],tuple(r["allowed_singleton_views"])))
    b={"allowed_singleton_views":tuple(sorted(x["allowed_singleton_views"])),"minimum_cross_pair_gap":x["minimum_cross_pair_gap"],"maximum_within_operand_endpoint_spread":x["maximum_within_operand_endpoint_spread"],"before_minimum_interval_span":x["before_minimum_interval_span"],"after_minimum_interval_span":x["after_minimum_interval_span"],"training_examples":len(rows),"training_authorizations":x["authorizations"],"training_wins":x["wins"],"training_losses":x["losses"],"training_ties":x["ties"],"selection_authority":"EXHAUSTIVE_3072_RULE_RISK_FIRST_FINITE_CLASS"};r=AsymmetricRule(**b,rule_sha256=stable_hash(b));r.validate();return r,tuple(c)
def apply_asymmetric_rule(binding,rule):
    binding.validate();rule.validate()
    if binding.authorized_relation is None:return binding
    gap,spread=binding_geometry(binding);view=singleton_view_kind(binding);support=maximum_interval_span(binding);required=rule.before_minimum_interval_span if binding.authorized_relation=="before" else rule.after_minimum_interval_span;reason=None
    if view is not None and view not in set(rule.allowed_singleton_views):reason=f"SOURCE_ABSTAIN_SINGLETON_VIEW_UNQUALIFIED:{view}"
    elif gap<rule.minimum_cross_pair_gap:reason=f"SOURCE_ABSTAIN_CROSS_PAIR_GAP_UNQUALIFIED:{gap}"
    elif spread>rule.maximum_within_operand_endpoint_spread:reason=f"SOURCE_ABSTAIN_ENDPOINT_SPREAD_UNQUALIFIED:{spread}"
    elif support<required:reason=f"SOURCE_ABSTAIN_RELATION_SUPPORT_UNQUALIFIED:{binding.authorized_relation}:{support}"
    if reason is None:return binding
    x=replace(binding,authorized_relation=None,reason=reason,receipt_sha256="");b=asdict(x);b.pop("receipt_sha256");x=replace(x,receipt_sha256=stable_hash(b));x.validate();return x
def asymmetric_target_grounder_sha256(**kwargs):return stable_hash({"schema_version":"agqa2-asymmetric-support-grounder-v55",**kwargs,"runtime_authority":"ABSTENTION_ONLY","outcome_or_label_runtime_input":False})
