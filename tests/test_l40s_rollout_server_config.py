from __future__ import annotations

from pathlib import Path

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
    assert command[command.index("--served-model-name") + 1] == "Qwen/Qwen3.5-9B"
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert command[command.index("--max-num-seqs") + 1] == "32"
    assert command[command.index("--max-num-batched-tokens") + 1] == "8192"


def test_formal_slurm_steps_partition_resources_without_overlap() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    batch = (repo_root / "cluster/run_principled_alfworld_2x4.sbatch").read_text()
    step_lines = [line.strip() for line in batch.splitlines() if line.lstrip().startswith("srun ")]
    assert len(step_lines) == 10
    assert all("--exact" in line and "--mem=0" in line for line in step_lines)
    assert all("--overlap" not in line for line in step_lines)
    assert '--gres="${GPU_GRES}:4" -N1 -n1 -c4' in batch
    assert batch.count('--gres="${GPU_GRES}:2" -N1 -n1 -c6') == 2
    assert 'GPU_GRES="${GPU_GRES:-gpu:l40s}"' in batch
    assert 'QUANTIZATION="${PROPOSER_QUANTIZATION}"' in batch
    assert 'elif [[ "${GPU_GRES}" == "gpu:rtxa6000" ]]' in batch
    assert 'VLLM_DISABLE_PYNCCL="${PROPOSER_PCIE_SAFE_COLLECTIVES}"' in batch
    assert 'scripts/verify_cuda_devices.py" 4' in batch
    assert "--gres=none -N1 -n1 -c1" in batch


def test_35b_server_can_disable_custom_allreduce_for_pcie_nodes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    server = (repo_root / "inference/serve_qwen35_35b_a3b.sh").read_text()
    assert 'DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"' in server
    assert 'VLLM_CMD+=( --disable-custom-all-reduce )' in server
    assert '--served-model-name "$SERVED_MODEL_NAME"' in server


def test_a6000_submitter_selects_safe_runtime_profile() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    submitter = (repo_root / "cluster/submit_principled_alfworld_a6000.sh").read_text()
    assert "--nodes=2" in submitter
    assert "--gres=gpu:rtxa6000:4" in submitter
    assert "--mem=120G" in submitter
    assert "GPU_GRES=gpu:rtxa6000" in submitter
