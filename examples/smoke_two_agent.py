from motif_transfer.contracts import Advisory, AdvisoryVerdict, Observation
from motif_transfer.decision_agent import FirstNativeDecisionAgent
from motif_transfer.runtime import TwoAgentRuntime


class AdmitOnlyMotifAgent:
    def propose_motifs(self, receipts):
        return ()

    def initialize_binding(self, motif, adaptation_receipts):
        return None

    def review(self, proposal, observation, binding, history):
        return Advisory(AdvisoryVerdict.ADMIT, "smoke")


class CounterEnvironment:
    def __init__(self):
        self.value = 0

    def reset(self):
        self.value = 0
        return Observation({"value": 0}, ("increment",))

    def step(self, action):
        assert action == "increment"
        self.value += 1
        done = self.value == 2
        return Observation({"value": self.value}, ("increment",), done, done, float(done)), float(done)


if __name__ == "__main__":
    runtime = TwoAgentRuntime(FirstNativeDecisionAgent(), AdmitOnlyMotifAgent())
    result = runtime.run(CounterEnvironment(), "reach two")
    print({"steps": len(result.receipts), "success": result.final_observation.official_success})
