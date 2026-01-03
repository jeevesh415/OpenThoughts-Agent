# Run DCFT experiments on a supercomputer

## Contents

- [Justification ](#justification)
- [Current work-around](#work-around)
- [Details on the launcher script](#launcher-script)
    - [Experiment config](#experiment-config)
    - [Google Sheets integration](#google-sheets-integration)
    - [Details on uploading to DB & HF](#upload-to-db--hf)
    - [Sync with wandb](#sync-with-wandb)
    - [Remote datasets & models](#remote-models--datasets)
    - [Restart a job](#automatically-restart-slurm-job-after-reaching-wall-time-limit)
    - [Details on evaluation](#details-on-evaluation)
        - [Using evalchemy for evals](#evaluation-with-evalchemy)
        - [Using eval config](#using-eval-config)
        - [Mulit-gpu evaluation](#multi-gpu-evaluation)
        - [Discrepancies between eval scores across different setups](#discrepancy-between-eval-scores-on-different-machinessetups)
        - [Check for already performed evaluations](#check-for-already-performed-evaluations)
- [Possible issues](#possible-future-issues)


## Justification

Supercomputers are different from cloud based clusters.
Main differences are following:

- **No internet connection on compute nodes.** It means that all the data, model checkpints, wandb and DB logging should be downloaded/uploaded outside of a compute node.

- **Slurm-based workflows.** Supercomputing facilities use [Slurm](https://slurm.schedmd.com/documentation.html) to schedule and mange jobs. It means that for every training/evaluation job you should create an sbatch script with specified resources (number of nodes, GPUs, CPUs etc) and accounting (account name, partition, time limit etc).

- **Strict storage rules.** On supercomputers there's always a limit on how many Inodes and/or bytesb you can store in a given project's HOME, SCRATCH, DATA etc. For example, you cannot use default directories for cache (~/.cache) because you will quickly eaxhoust disk quota of your project and will not only block your own workload but the whole projects that other people use can be blocked as well.

- **No sudo access.** You cannot install any software on a supercomputer that requires sudo access. For example, you cannot run docker container of a supercomputer. However, you can build singularity (apptainer) images from a docker image to run your custom software. You also can build/use modules.

## Work-around

- **No-internet related issues.** Although there's no internet connection on the main compute nodes, there's internet connection on the devel(opment) partions. You cannot use more than 4 devel nodes at a time and your job cannot last more than 2 hours, so you cannot run your actual job on devel partition (and it's highly discouraged to do so, you can run low-resource jobs like evaluations if these jobs fit resource constraints). However you can run your train job on main compute nodes and then schedule a job on devel nodes that will depend on the compute job, i.e. only if the main job finishes successfully, the job that uploads data to HF/DB/whatever will be scheduled. For the downloading issue the solution is to doenload data/models on login/devel nodes to some directory and then load them from there (see more info below in the *Strict storage rules* section).

- **Need of sbatch script.** One approach is to construct sbatch scripts automatically by using some kind of base sbatch script/config combination. In the config file you can specify stuff related to the supercomputer/slurm - account, partition, number of nodes etc. The base sbatch scripts is just a template where ypu can insert your desired parameters.

Condider example below.

Template sbatch file:
```bash
#!/bin/bash -x
#SBATCH --nodes={num_nodes}
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --threads-per-core=1
#SBATCH --time={time_limit}
#SBATCH --output={output_dir}/logs/%x_%j.out
```
Config:
```yaml
base_sbatch_script_path: dcft/train/jsc/train_base.sbatch
num_nodes: 4
partition: dc-gpu-devel
account: cbrdg24
time_limit: 02:00:00
```

The sbatch file that will run:
```bash
#!/bin/bash -x
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --account=cbrdg24
#SBATCH --partition=dc-gpu-devel
#SBATCH --threads-per-core=1
#SBATCH --time=02:00:00
#SBATCH --output=DCFT_experiments/logs/%x_%j.out
```

See here examples of DCFT [train](batch_templates/train_base.sbatch) and [eval](batch_templates/train_base.sbatch) templates.

- **Strict storage rules.** For this we can specify environment variables for directories that (will) store cache, model checkpoints, datasets:

```bash
export CHECKPOINTS_DIR="" # directory that checkpoints are saved to
export MODELS_DIR="" # directory that models are downloaded to
export DATASETS_DIR="" # directory that datasets are downloaded to including dataset_info.json
export HF_HOME="" # specify cache directory
```

And everytime we will load (download) a model or a dataset, these paths will be used instead of default ones.
This is also related to the first problem of no internet connection (see above).

- **No sudo access.** If you want run something that requires sudo access, you can use [apptainer](https://apptainer.org/documentation/).

## Launcher script

Launcher is the utility script that allows you to construct and submit sbatch scripts.
Currently it has two main functionalities: `main` and `collect_results`.
The main functionality requires you to provide path to your [experiment config](#experiment-config) YAML filename. See [this example](run_exp.sh) for more info.

```bash
exp_config="$DCFT_PRIVATE/dcft/train/jsc/exp_configs/base.yaml" # experiments config
LAUNCHER_SCRIPT="$DCFT_PRIVATE/dcft/train/jsc/launcher.py"
FUNC="main"
CMD="python $LAUNCHER_SCRIPT $FUNC --exp_config_path $exp_config"
bash -c $CMD
```

### Experiment config

The experiment config is a YAML file. Here is an [example](exp_configs/base.yaml) of such a config.
The main required arguments are following:
```yaml
base_sbatch_script_path: # path to the template train sbatch script
base_config_path: # Llama-Factory YAML config (or a base config)*
output_dir: # directory to save logs, constructed sbatch scripts etc
train_args: [] # arguments that the job name will be constructed from and that will be inserted into train sbatch template, e.g. base_model, dataset etc)
num_nodes: # number of nodes for train jobs*
partition: # compute partition*
account: # compute account*
time_limit: # time limit of running the job*
```
*you can use a remote Google sheet to specify all the paremeters (both supercomputer specific and training arguments). The principle is the same as with template sbathc scripts: ypu should provide template YAML config that the values will be inserted into. More info in the [corresponding section](#google-sheets-integration).

For evaluation it's similar:
```yaml
eval_args: [model_save_dir] # directory where model weights are saved
eval_base_script_path: dcft/train/jsc/eval_base.sbatch
num_nodes_eval: 1
partition_eval: dc-gpu-devel
account_eval: cbrdg24
time_limit_eval: 02:00:00
eval_tasks: "mmlu" # for now, lm eval harness tasks, use commma as separator for multiple tasks
```
And for upload:
```yaml
upload_base_script_path:
upload_args: [model_save_dir]
account_upload: cbrdg24
partition_upload: dc-cpu-devel
time_limit_upload: 00:30:00
```


### Google Sheets integration

See [example](exp_configs/remote.yaml) of experiment config that uses remote google sheet. You will need to setup your Google Clud credentials and install required Python packages. See [this guide](https://developers.google.com/docs/api/quickstart/python).

The launcher will automatically take all the values from each column and for each row construct sbatch scripts, save them to specified output directory and submit jobs. Then it will save arguments like `model_save_dir`, `job_name`, `satus` etc.

Instead of LF YAML config we use a base config to insert value into, see example of such a spreadsheet [here](https://docs.google.com/spreadsheets/d/1UbERJWcylk-6ggeteoNCVan_hZqHnxZOQhEkoDns0Bg/edit?usp=sharing).

###  Upload to DB & HF

Since we don't have internet connection on superomputer compute nodes, we need to use either login or devel nodes. The upload should be done before the evaluation of the model (becuase during evals we need to take model info from HF). We use [upload_local.py](../upload_local.py) helper script to upload model weights to HF and experiment info to DB. See the [upload template](sbatch_templates/upload_base.sbatch) sbatch script.

### Sync with Wandb

Since we are using wandb offline mode, we can save the logs localy and then sync them up with the wandb server. It can be done in to steps:
1. Launcher will create and save wandb logs to specified the `<OUTPUT_DIR>/wandb`. On the compute node the logs will be saved locally.
2. The [upload_local.py](../upload_local.py) script will take all the saved logs and sync it up with wandb.ai.

Usefull wandb [environment variables](https://docs.wandb.ai/guides/track/environment-variables/):
```bash
export WANDB_CACHE_DIR="anndb_cache" # set the cache directory
export WANDB_PROJECT="dcft"
```

### Remote models & datasets

In the LLama factory config file, you can specify HF hub IDs for both remote model and dataset (as well as specific arguments). The launcher will check if a dataset (or model) exists inthe `DATASETS_DIR` (or `MODELS_DIR`) and download them ther if they are not.
Useful envirnonment variables:

```bash
export MODELS_DIR="models/" # directory that models are downloaded to
export DATASETS_DIR="lf_datasets/" # directory that datasets are downloaded to 
```
**Important note:** make sure that both `MODELS_DIR` and `DATASETS_DIR` exist, have enough space and they are set properly, otherwise you models and datasets will be downloaded to you home dir (`~`) which can result in reaching of disk quota or will not be able to be downloaded!

Example:
```yaml
model_name_or_path: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dataset: tatsu-lab/alpaca
dataset_dir: ONLINE
template: alpaca
formatting: alpaca
```

### Automatically restart Slurm job after reaching wall time limit

Because supercomputers use Slurm to schedule jobs, ever has it's time limit (`--time`). Sometimes the training job cannot be finished in this period and you need to restart. To make it less manual, you can provide `max_restarts`(integer, default is 0) argument into the experiment config. The job will be submitted `max_restarts + 1` times in such a way that when the previuos job gets finished, the next job will be scheduled.
Example:
 ```yaml
 max_restarts: 2
 ```
 In the example above the whole training time will be equal to `3 * time_limit`.

 **Importanta note:** a job can be successfully restarted only if a checkpoint was saved successfully.

### Details on evaluation
#### Evaluation with evalchemy
 We use [evalchemy](https://github.com/mlfoundations/evalchemy/tree/main) to perform evaluations. Clone the repo and set the corresponding environment variable to the path ypu cloned it to:
```bash
export EVALCHEMY_HOME="/your/full/path/evalchemy"
``` 
#### Using eval config
 In order to perform evaluation on a set of tasks using an eval config file, in the experiment config file you can provide your eval config, e.g:
 ```yaml
 eval_config_path: /your/full/path/evalchemy/configs/light_gpt4omini0718.yaml
 ```
 Make sure to set you batch sizes inside the config to theones that will fit in your hardware (it might be different from defaults).

#### Multi-gpu evaluation
 By default the [eval_base.sbatch](sbatch_templates/eval_base.sbatch) template is set to run on one node with mulitple GPUs (using data-parallel regime). See more info [here](https://github.com/mlfoundations/evalchemy/tree/main?tab=readme-ov-file#multi-gpu-evaluation).

#### Discrepancy between eval scores on different machines/setups
 In case some evaluation are significantly different from evaluations performed on other machines, make sure to check your environment. You can build a new environment from scratch by following [these instructions](https://github.com/mlfoundations/evalchemy/tree/main?tab=readme-ov-file#installation).

#### Check for already performed evaluations
 In the [launcher.py](launcher.py) before submitting jobs, we check if a model was trained and uploaded to HF already (so we don't train/upload it again) as well as what eval tasks were already done. To do this we check all the result files in the `{model_save_dir}/eval` directory and compare already existing tasks to the requested tasks. Then we only perform the tasks that were not done yet. To still rerun the evals you can set the `redo_evals` to true in your exp config file:
```yaml
redo_evals: true
```

## Possible (Future) Issues

1. (Update: it's solved. If the model was already uploaded and nothing has changed - we don't upload anything and the `git_commit_hash` stays the same) 
When uploading model to HF, we check whether model repo exists on HF hub and if it doesn't, we upload the model weights. However it's not ideal since model weights might change and we woild like to have an option to overwrite it. 

2. (Update: solved) We need to normalize model and dataset names (the base models and datasets are comming from HF hub and they are in the form of "org_name/model_name"). We cannot load both of them online and need to load them from local folders. The main problem arises from output directories/pathes contstructions - we cannot have "/" in hf_hub_id for example (except one). Also it's important for log filenames. The problem is when we try then to upload this model to the database, it tries to get base model/dataset info from HF hub which will not work with normalized name. There're might be some issues related to this.

3. Time-specification in batch files `-t` can be wrongly parsed when it is in hh:mm:ss (it works correctly in 00:mm::ss format). It is better to put it in quotes on JSC. It led to `QOSMaxWallDurationPerJobLimit` error without quotes.
