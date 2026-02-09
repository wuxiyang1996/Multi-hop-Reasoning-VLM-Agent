# Setting up the evolver_eval conda environment

Follow the **AgentEvolver games** instructions so the `evolver_eval` env can run Avalon/Diplomacy and evaluation.

## 1. Create the conda environment (if not already created)

From the **codebase root** (`D:\ICML_2026\codebase`):

```powershell
conda create -n evolver_eval python=3.11 -y
```

## 2. Install game dependencies (per games/README.md)

Activate the env and install the minimal requirements for the games web UI and evaluation:

```powershell
conda activate evolver_eval
pip install -r AgentEvolver\games\requirements_game.txt
```

Or in one shot without activating:

```powershell
conda run -n evolver_eval pip install -r AgentEvolver\games\requirements_game.txt
```

## 3. (Optional) Environment variables for LLM APIs

```powershell
$env:OPENAI_BASE_URL = "your_api_url"
$env:OPENAI_API_KEY = "your_api_key"
```

## 4. Run evaluation tests

From codebase root with `evolver_eval` activated:

```powershell
conda activate evolver_eval
$env:PYTHONPATH = (Get-Location).Path + ";" + (Get-Location).Path + "\AgentEvolver"
python evaluation_evolver\test_avalon_dummy.py --max_phases 20
python evaluation_evolver\test_diplomacy_dummy.py --max_phases 3
```

For Diplomacy you may also need the AI_Diplomacy dependency:

```powershell
pip install coloredlogs
```
