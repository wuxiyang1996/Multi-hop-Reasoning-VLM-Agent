from __future__ import annotations

from trainer.coevolution.vllm_server import VLLMServerManager


def test_l40s_context_and_batch_limits_reach_vllm_command(tmp_path) -> None:
    manager = VLLMServerManager(
        model_name="Qwen/Qwen3.5-9B",
        adapter_dir=str(tmp_path),
        gpu_ids=[0],
        max_model_len=8192,
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        speculative_method="none",
        reasoning_parser=None,
    )
    command = manager._build_serve_cmd(8000, [], False)
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert command[command.index("--max-num-seqs") + 1] == "32"
    assert command[command.index("--max-num-batched-tokens") + 1] == "8192"
