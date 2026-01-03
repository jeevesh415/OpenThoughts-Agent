# DC-Agent: Data Recipes for Training Agentic Models

Welcome to DC-Agent, a large-scale research project dedicated to creating the best tooling and finding the best data for training small agentic models.

## Links

Project Website: Coming Soon

[DC-Agent Leaderboard](https://dcagents-leaderboard.replit.app/)

[DC-Agent Trace Viewer](https://dcagents-trace-viewer.replit.app/)

## Warning!

DC-Agent is a research codebase! Conventions will change, files will move and workflows will break as we continue to grow. Please bear with us and open an issue if you discover a bug.

## Getting Started

If you are new to the project, start here to get up and running.

### Installation

From the root directory, you can install the core HPC + data infrastructure dependencies with:

`pip install .`

* **SFT stack**:  
  * First ensure the git submodule is initialized:  
    `git submodule update --init --remote dcft/train/llamafactory`  
  * Then install SFT dependencies:  
    `pip install .[sft]`
  * The submodule itself also defines optional extras (e.g. `liger-kernel`, `deepspeed`, `hf-kernels`) in its own `pyproject.toml`; install those as needed after `pip install .[sft]`.
  * For more information on the SFT stack, check out:
    `dcft/train/llamafactory/README.md`
* **Data stack**
  * Data helper tools (CLI/data-processing utilities used in some datasets):  
    `pip install .[data]`
  * For more information on the data stack, check out: `data/README.md`

### Secrets and API Keys

Most scripts expect credentials (HF tokens, Daytona keys, W&B API keys, Supabase creds, etc.) to live in a private `env` file that is **not** committed to this repo. Point DC-Agent at your private file by exporting:

```bash
export DC_AGENT_SECRET_ENV=/secure/path/to/my_dc_agent_secrets.env
```

That file should `export DAYTONA_API_KEY=...`, `export HF_TOKEN=...`, `export WANDB_API_KEY=...`, `export SUPABASE_*`, etc. The launcher and auxiliary scripts now read `DC_AGENT_SECRET_ENV`; legacy `KEYS`/`SECRET_ENV_PATH` variables are still accepted for backward compatibility but will be removed once everyone migrates.

### Launching a Job

DC-Agent's job launchers are designed to work with HPC (high-performance computing) clusters. Different launchers exist for different job types. DC-Agent's launchers are modular, making it relatively straightforward to add your own preferred cluster.

#### How to Launch a Datagen Job

Datagen jobs are launched via the generic HPC launcher and use `--job_type datagen` plus a generator script.

1. Ensure your cluster environment is set up (dotenv, conda env, etc.). For TACC, see `hpc/docs/TACC.md` and `hpc/dotenv/tacc.env`.
2. Activate your environment and source the dotenv:
   ```bash
   source hpc/dotenv/<your-cluster>.env
   $DCFT_ACTIVATE_ENV
   cd "$DCFT"
   ```
3. Choose or write a datagen script under `data/...` implementing `BaseDataGenerator` (see `data/generation.py` and existing generators for examples).
4. Run the launcher from a login node:
   ```bash
   python -m hpc.launch \
     --job_type datagen \
     --datagen_script data/<dataset>/generate.py \
     --datagen_target_repo <org/dataset-tasks> \
     --datagen_engine vllm_local \
     --datagen_extra_args "--stage both --limit 200" \
     --experiments_dir "$DCFT/experiments" \
     --time_limit 12:00:00
   ```
5. To also generate traces, add:
   - `--enable_trace_gen`  
   - `--trace_target_repo <org/dataset-traces>`  
   - `--trace_harbor_config path/to/harbor_job.yaml`  
   and any of the `trace_*` overrides documented in `hpc/README.md`.

The launcher will synthesize and submit one or more `sbatch` scripts under `"$experiments_dir/sbatch_scripts"` and write configs to `"$experiments_dir/configs"`. Use `--dry_run` to inspect scripts without actually calling `sbatch`.

#### How to Launch an SFT Job

SFT jobs are also launched via `hpc.launch` with `--job_type train` and a LLaMA Factory config.

1. Pull and install the SFT submodule (once per checkout):
   ```bash
   git submodule update --init --remote dcft/train/llamafactory
   pip install .[sft]
   ```
2. Configure your cluster dotenv and environment as in the Datagen section.
3. Pick a training config under `dcft/train/hp_settings` or create your own YAML.
4. From a login node, run:
   ```bash
   python -m hpc.launch \
     --job_type train \
     --train_config_path dcft/train/hp_settings/<path-to-config>.yaml \
     --dataset <org/dataset> \
     --num_nodes 8 \
     --time_limit 24:00:00 \
     --experiments_dir "$DCFT/experiments"
   ```
5. Optionally override LLaMA Factory flags via `--train_extra_args "..."` (see `hpc/README.md` and `dcft/train/llamafactory/README.md` for full argument lists).

The launcher will construct a per-run YAML in `"$experiments_dir/configs"`, generate an sbatch script, and then submit the job. Training metadata and summaries are written into the run’s `output_dir`.

#### How to Launch an RL Job

RL training currently uses cluster-specific scripts under `rl/` rather than the generic `hpc.launch` entry point.

1. Make sure you have access to the shared RL environment and Ray/vLLM backend described in the TACC docs and comments inside `rl/tacc/tacc_train_rl_tbench.sh`.
2. Log into the target cluster (e.g., TACC Vista) and load the required modules (CUDA, Apptainer, GCC) as shown in the script.
3. Edit `rl/tacc/tacc_train_rl_tbench.sh` to point to your:
   - data directories
   - checkpoint/output paths
   - base model ID (e.g. `Qwen/Qwen2.5-7B-Instruct`)
   - sandboxes / trace storage locations
4. From a login node, submit the job:
   ```bash
   sbatch rl/tacc/tacc_train_rl_tbench.sh
   ```
5. Monitor logs under the `experiments/logs` directory configured in the script and resume/tune hyperparameters via the `skyrl_train.entrypoints.main_base` arguments inside the sbatch file.

#### How to add your cluster to DC-Agent

Adding a new cluster involves defining its resources, sbatch templates, and a dotenv file so `hpc.launch` can target it.

1. **Create a dotenv for your cluster** under `hpc/dotenv/`, following `tacc.env` as a template. At a minimum, define:
   - `DCFT` (path to your dc-agent checkout on the cluster)
   - `DCFT_ACTIVATE_ENV` (command to activate the Python env)
   - paths for `EXPERIMENTS_DIR`, `DATASETS_DIR`, `MODELS_DIR`, and any cluster-specific SIF/Apptainer images.
2. **Register basic cluster metadata** by exporting `HPC_NAME` and related fields in your dotenv or by passing them on the CLI:
   - `--name`, `--account`, `--partition`, `--gpus_per_node`, `--cpus_per_node`, etc. (see `hpc/README.md` and `hpc/hpc.py`).
3. **Create sbatch templates** in `hpc/sbatch_data/` for your cluster:
   - Copy an existing template for a similar machine (GPU type / internet access) and adjust `#SBATCH` headers and module loads.
   - Keep placeholders like `{time_limit}`, `{job_name}`, `{experiments_dir}` etc. intact; they will be filled by `hpc.launch`.
4. **Declare required templates** in `hpc/sbatch_data_requirements.json` so `_validate_sbatch_templates` can verify your cluster has all needed sbatch files for datagen and training.
5. **Test with a dry run**:
   ```bash
   source hpc/dotenv/<your-cluster>.env
   $DCFT_ACTIVATE_ENV
   cd "$DCFT"
   python -m hpc.launch \
     --job_type datagen \
     --datagen_script data/<dataset>/generate.py \
     --datagen_target_repo test-org/test-dataset \
     --experiments_dir "$DCFT/experiments" \
     --dry_run
   ```
6. Once sbatch scripts look correct, drop `--dry_run` to submit real jobs. If your cluster needs special handling (login vs compute nodes, proxies, etc.), add it to `hpc/hpc.py` and, if necessary, `hpc/launch.py` (for example, see the existing logic for JURECA/JUWELS internet nodes).

#### Learn More about HPC Launch

To learn more about the details of how HPC Launch works, please refer to `hpc/README.md`.

### Who to contact if you get stuck

Please reach out to someone on the [terminal-bench Discord](https://discord.gg/6xWPKhGDbA) if you need help.

* For RL: Please contact Charlie Ruan
* For SFT: Please contact Benjamin Feuer
* For Data: Please contact Etash Guha
* For Eval: Please contact Negin Raoof
* For Project Management (includes cluster and account access): Please Contact Ryan Marten

## DC-Agent is Built On

[Llama Factory](https://github.com/hiyouga/LLaMA-Factory)

[SkyRL](https://github.com/NovaSky-AI/SkyRL)

[vLLM](https://github.com/vllm-project/vllm)

[Harbor](https://github.com/laude-institute/harbor)

## Friends of DC-Agent

[![Daytona Startup Grid](https://img.shields.io/badge/SPONSORED%20BY-DAYTONA%20STARTUP%20GRID-2ECC71?style=for-the-badge)](https://daytona.io/startups?utm_source=datacomp.ai)

[Laude Institute](https://www.laude.org/)

[Bespoke Labs](https://www.bespokelabs.ai/)

[Oumi](https://oumi.ai/)

## Citation

Coming Soon
