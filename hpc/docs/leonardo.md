# CINECA Leonardo - DCFT Training Guide

## Quickstart

First ensure these are in your `~/.bashrc`
```
source /leonardo_work/EUHPC_E03_068/DCFT_shared/dcft_private/hpc/dotenv/leonardo.env
source $DCFT_PRIVATE/hpc/scripts/common.sh
```

Training and uploading are done in separate steps, since the nodes on Leonardo do not have internet. 

### Training 
**MAKE SURE YOUR HP SETTING IS CORRECT! (micro, small, medium, large)**. Check the [Google Sheet](https://docs.google.com/spreadsheets/d/11ThWrGsEpT56Hxa_C3JyVEP33lt2V2fz1XSu0kcXdo8/edit?gid=633170063#gid=633170063), it will say. Note that when using 128 nodes (512 GPUs) the batch size needs to be divisible by 512 (this is only true for large - where GBS is 512). 

```bash
python3 -m hpc.launch --train_config_path dcft/train/hp_settings/paper/reasoning_large.yaml --time_limit 1-00:00:00 --num_nodes 128 --dataset mlfoundations-dev/openthoughts2 --qos boost_qos_bprod --pretokenize --max_restarts 3
```

- This trains the model called "openthoughts2".
- Check that the train config (e.g. `reasoning_large.yaml`) is appropriate for the model.
- For now, we want to always run on 128 nodes and only large runs that use reasoning_large.yaml with batch size 512. This is because the current queueing priority system puts `boost_qos_bprod` on top, and this qos needs a minimum of 128 nodes.
- `pretokenize` will launch a 30min job with the `boost_qos_dbg` qos that will create the tokenized dataset on a single node and write to `$TOKENIZED_DATASETS_DIR`. This avoids the synchronization issues with the filesystem when trying to do the tokenization on the multinode setup. 
- `max_restarts` allows you to train for more than 1 day (1-00:00:00) by automatically restarting the job when it "fails" (aka TIMEOUT). Llamafactory will continue from the last checkpoint



### Uploading
Once training is completed, upload your model. (See the FAQ for more [info](https://github.com/mlfoundations/dcft_private/blob/team/leonardo/hpc/docs/leonardo.md#how-do-i-upload-a-model-) and how to [automate](https://github.com/mlfoundations/dcft_private/blob/team/leonardo/hpc/docs/leonardo.md#how-do-i-automate-uploading-models-after-training-completes-) this)
If you `source $DCFT_PRIVATE/hpc/scripts/common.sh` in your `~/.bashrc` just check to see completed jobs over the last `N` hours
```
swin N
```
And upload jobs that are done
```
upload openthoughts2
```

## Cluster Information and FAQ

Start by reading the [Get-Started on the official documentation](https://wiki.u-gov.it/confluence/display/SCAIUS/Get+Started). Many of your other questions can be answered there. In this document we detail information used frequently for DCFT training. Other resources include Jenia's [guide on Leonardo](https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig) which details basic setup, environments, singularity, and data transfer and Marianna's documentation on [DCFT for JSC](https://github.com/mlfoundations/dcft_private/tree/main/dcft/train/jsc) which has a similar setup to Leonardo. 


<details>
<summary><h3>How do I join the cluster?</h3></summary>

See [Jenia's instructions](https://iffmd.fz-juelich.de/e-hu5RBHRXG6DTgD9NVjig#Basic-info) and the [official instructions](https://wiki.u-gov.it/confluence/display/SCAIUS/Get+Started). 

⚠️ **WARNING:**
1. Don't forget to **message Jenia your username**
2. Don't let your final **approval expire after 1 day** of recieving the email
</details>

<details>
<summary><h3>How do I automate the certificate checking?</h3></summary>

Once you set it up manually, you can use this nifty script. You also need to setup a `leonardo` entry in your `~/.ssh/config`. And you can add this script as an alias to your local `~/.bashrc`. 

Update `export EMAIL=<YOUR-EMAIL=HERE>` to be the one you signed up for Leonardo with. 
 
```
#!/bin/bash

# Script to automate Cineca HPC login

# Check if certificate is valid or expired

export EMAIL=<YOUR-EMAIL-HERE>
echo "Checking if certificate is valid..."
cert_info=$(step ssh list --raw $EMAIL | sed -n '2p' | step ssh inspect 2>/dev/null)

if [ $? -ne 0 ] || ! echo "$cert_info" | grep -q "Valid:" || ! echo "$cert_info" | grep -q "to 20"; then
    echo "Certificate not found or format unexpected. Obtaining new certificate..."
    step ssh login $EMAIL --provisioner cineca-hpc
elif echo "$cert_info" | grep -q "Valid:"; then
    # Extract expiry date and time
    expiry=$(echo "$cert_info" | grep "Valid:" | sed -E 's/.*to ([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}).*/\1/')

    # Convert dates to seconds since epoch for comparison
    expiry_seconds=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$expiry" +%s 2>/dev/null)
    current_seconds=$(date +%s)

    if [ -z "$expiry_seconds" ] || [ $current_seconds -ge $expiry_seconds ]; then
        echo "Certificate expired. Obtaining new certificate..."
        step ssh login $EMAIL --provisioner cineca-hpc
    else
        echo "Certificate is still valid until $expiry."
    fi
fi

# Connect to cineca
echo "Connecting to Cineca HPC..."
ssh leonardo
```
</details>


<details>
<summary><h3>Where should I be storing my code / data / models?</h3></summary>

For DCFT, we store everything in a shared folder in `$WORK/DCFT_shared`. These shared installations should be kept in working order for quick training and eval launches. More experimental work can be done by cloning your own copy of evalchemy / dcft_private into `$WORK/<YOUR-USERNAME>`.

You can conveinently store relevant DCFT locations by adding environment variables to your `~/.bashrc`. 

```
cat << 'EOF' >> ~/.bashrc
export DCFT=$WORK/DCFT_shared
export DCFT_MAMBA=$DCFT/mamba
export DCFT_PRIVATE=$DCFT/dcft_private
export DCFT_PRIVATE_ENV=$DCFT_PRIVATE/env/dcft-private
export EVALCHEMY=$DCFT/evalchemy/
export EVALCHEMY_GPU_ENV=$EVALCHEMY/env/cu121-evalchemy
export EVALCHEMY_CPU_ENV=$EVALCHEMY/env/cpu-evalchemy
export HF_HUB_CACHE=$DCFT/hub
EOF
source ~/.bashrc
```

- `$DCFT` is the shared folder for all things DCFT related
- `$DCFT_PRIVATE` is the shared git repository for dcft_private
- `$DCFT_PRIVATE_ENV` is the shared conda environment for dcft_private 
- `$EVALCHEMY`is the shared git repository for evalchemy
- `$EVALCHEMY_GPU_ENV` is the shared conda environment for evalchemy that works on the GPU node
- `$EVALCHEMY_CPU_ENV` is the shared conda environment for evalchemy that works on the login node
- `$HF_HUB_CACHE` is the shared huggingface cache models and datasets

The official Leonardo documentation describes more in detail the storage options and the suggested usage for each. 
- [Quick Start (6. Storage)](https://wiki.u-gov.it/confluence/display/SCAIUS/Get+Started#:~:text=and%20software%20installations.-,6.%20Storage,-Our%20HPC%20systems)

> Our HPC systems offer several options for data storage:
>
> - $HOME: to store programs and small light results
> - $CINECA_SCRATCH: where you can execute your programs
> - $WORK: An area visible to all the users associated with the same budget account
> - $DRES: An additional area to store your results if they are heavy. This space is not automatic. You need to request for it writing to superc@cineca.it
>
> Important details and suggestions on how to use each space can be found on the ["1.4 Data Storage and Filesystem" page](https://wiki.u-gov.it/confluence/display/SCAIUS/4%3A+Data+storage+and+FileSystems).
> 
> To monitor the occupancy of your space, you can use the ["cindata"](https://wiki.u-gov.it/confluence/display/SCAIUS/4%3A+Data+storage+and+FileSystems#id-4:DatastorageandFileSystems-Monitoringtheoccupancy) command.

There is also a [summary](https://wiki.u-gov.it/confluence/display/SCAIUS/LEONARDO+User+Guide#:~:text=for%20further%20informations.-,Disks%20and%20Filesystems,-The%20storage%20organization) in the main user guide. 

</details>


<details>
<summary><h3>How does offline model and dataset handling work?</h3></summary>

The DCFT training infrastructure supports working in offline mode, where models and datasets are accessed from local cache rather than downloaded from Hugging Face.

- Models are downloaded via the [download_model()](https://github.com/mlfoundations/dcft_private/blob/873c086ce504dfbf03fb37e8624f8ac22679a715/dcft/train/leonardo/train.py#L272-L283) function
- Datasets are downloaded via the [download_dataset()](https://github.com/mlfoundations/dcft_private/blob/873c086ce504dfbf03fb37e8624f8ac22679a715/dcft/train/leonardo/train.py#L286-L297) function
- Download uses `huggingface_hub.snapshot_download` to retrieve and cache model files
- Download goes to `$HF_HUB_CACHE` or `/leonardo_work/EUHPC_E03_068/DCFT_shared/hub` by default
- The YAML config [sets datasets_cache_dir](https://github.com/mlfoundations/dcft_private/blob/873c086ce504dfbf03fb37e8624f8ac22679a715/dcft/train/leonardo/train.py#L330) to point to this cache
- `HF_HUB_OFFLINE=1` is set in the [sbatch template](https://github.com/mlfoundations/dcft_private/blob/873c086ce504dfbf03fb37e8624f8ac22679a715/dcft/train/leonardo/template.sbatch#L63)
- When set, the parser [skips database communication](https://github.com/mlfoundations/dcft_private/blob/873c086ce504dfbf03fb37e8624f8ac22679a715/dcft/train/llamafactory/src/llamafactory/data/parser.py#L150-L163)
- In LlamaFactory, datasets are loaded with [load_dataset(..., cache_dir=data_args.datasets_cache_dir)](https://github.com/mlfoundations/dcft_private/blob/873c086ce504dfbf03fb37e8624f8ac22679a715/dcft/train/llamafactory/src/llamafactory/data/loader.py#L121-L132) which uses the cache directory from the YAML config
- Training jobs automatically use cached models and datasets when running in offline mode

Debugging details on why this is the way to do it: Our version of llamafactory requires `datasets>=2.16.0,<=3.1.0`, as specified in [`check_dependencies()`](https://github.com/mlfoundations/dcft_private/blob/873c086ce504dfbf03fb37e8624f8ac22679a715/dcft/train/llamafactory/src/llamafactory/extras/misc.py#L75-86). The newer version of datasets allows for the `dataset_path` returned by `snapshot_download` to be passed as the only argument to `load_dataset`. The older version does not allow that. This works if you upgrade datasets `pip install -u datasets`. and disable the llamafactory version check `DISABLE_VERSION_CHECK=1` but I decided not to do this since the consequences are unknown. Instead, with this version, we just pass the repo id to `load_dataset` along with the `dataset_cache_dir`. However, *this still does not work*. In order for the older version of datasets to correctly find the dataset, the dataset needs to previously be successfully loaded. This is why in the [launch script](https://github.com/mlfoundations/dcft_private/blob/873c086ce504dfbf03fb37e8624f8ac22679a715/dcft/train/leonardo/train.py#L294) we `load_dataset` after `snapshot_download`. 


</details>

<details>
<summary><h3>How to check for node availability?</h3></summary>

```
sinfo -S+P -o "%18P %8a %20F"
```

This tells you the current usage of the cluster nodes. 

- **A**: Allocated - Nodes that are currently in use, running jobs
- **I**: Idle - Nodes that are available and ready to accept jobs
- **O**: Other - Nodes in other states (like mixed, draining, down, etc.)
- **T**: Total - Total number of nodes in that partition

You can make this easier for yourself by adding an alias `sinf` in your `~/.bashrc`:

```
echo "alias sinf='sinfo -S+P -o \"%18P %8a %20F\"'" >> ~/.bashrc
source ~/.bashrc
sinf
```

Example output

```
PARTITION          AVAIL    NODES(A/I/O/T)
boost_fua_dbg      up       3067/30/1/3098
boost_fua_prod     up       3067/30/1/3098
boost_usr_prod     up       3067/30/1/3098
dcgp_fua_dbg       up       1389/9/0/1398
dcgp_fua_prod      up       1389/9/0/1398
dcgp_usr_prod      up       1389/9/0/1398
lrd_all_serial*    up       2/0/0/2
lrd_cin_viz        up       0/15/0/15
```

The main partition we use is `boost_usr_prod`. Here you can see that 30 nodes (each with 4x A100 64GB) are currently idle. You can read more about the partitions in the [CINECA docs](https://wiki.u-gov.it/confluence/display/SCAIUS/Booster+Section#:~:text=partitions%20of%20LEONARDO%20Booster). I've copied the table below: 


| SLURM partition | Job QOS | # cores/# GPU per job | max walltime | max running jobs per user/max n. of nodes/cores/GPUs per user | priority | notes |
|-----------------|---------|------------------------|--------------|---------------------------------------------------------------|----------|-------|
| lrd_all_serial (default) | normal | max = 4 physical cores (8 logical cpus) max mem = 30800 MB | 04:00:00 | 1 node / 4 cores / 30800 MB | 40 | No GPUs Hyperthreading x2 |
| boost_usr_prod | normal | max = 64 nodes | 24:00:00 | | 40 | |
| | boost_qos_dbg | max = 2 nodes | 00:30:00 | 2 nodes / 64 cores / 8 GPUs | 80 | |
| | boost_qos_bprod | min = 65 nodes max = 256 nodes | 24:00:00 | 256 nodes | 60 | runs on 1536 nodes min is 65 FULL nodes |
| | boost_qos_lprod | max = 3 nodes | 4-00:00:00 | 3 nodes /12 GPUs | 40 | |
</details>

<details>
<summary><h3>How to check for my / someone else's / all queued runs?</h3></summary>
Check my queued runs:

```
sqme
```

or

```
squeue --me
```

or

```
squeue -u (my_username_here)
```

Check someone else's queued runs:
```
squeue -u (username_here)
```

Check all queued runs:
```
squeue
```

Here, what you want to look at is the "S" column. "P" means the job is pending. "R" means the job is running.
</details>

<details>
<summary><h3>How was the shared python environment originally created?</h3></summary>

For instructions on how the shared mamba environment was created see https://github.com/mlfoundations/evalchemy/blob/main/eval/distributed/SETUP_LEONARDO.md

```
# Set up environment variables in your .bashrc for easier access
cat << 'EOF' >> ~/.bashrc
export DCFT=$WORK/DCFT_shared
export DCFT_PRIVATE=$DCFT/dcft_private
export DCFT_MAMBA=$DCFT/mamba
export DCFT_PRIVATE_ENV=$DCFT_PRIVATE/env/dcft-private
export HF_HUB_CACHE=$DCFT/hub
EOF
source ~/.bashrc

eval "$(${DCFT_MAMBA}/bin/conda shell.${SHELL_NAME} hook)"
${DCFT_MAMBA}/bin/mamba create -y --prefix ${DCFT_PRIVATE_ENV} --clone base
source ${DCFT_MAMBA}/bin/activate ${DCFT_PRIVATE_ENV}

pip install -r dcft/train/requirements.txt
pip install -e .

# 3hr test run
python3 $WORK/DCFT_shared/dcft_private/dcft/train/leonardo/train.py --model Qwen/Qwen2.5-7B-Instruct --dataset bespokelabs/Bespoke-Stratos-17k --yaml $WORK/DCFT_shared/dcft_private/dcft/train/hp_settings/reasoning.yaml --nodes 4
```
</details>

## Training FAQ and common errors encountered
<details>
<summary><h3>How can I track my run?</h3></summary>
1. Run `sqme` to check that your run hasn't crashed yet. You can also join #dcft-slurm-notifs on Slack to be notified of crashes.
2. Look for your logs in `/leonardo_work/EUHPC_E03_068/DCFT_shared/dcft_private/experiments/logs`. The filename pattern is generally `{dataset}_{jobid}.out`
</details>

<details>
<summary><h3>How can I debug my crashed run?</h3></summary>
There's no quick answer to this. The easiest way is to view logs. (see question above). 

For common errors, we list some of them below. If not yet solved, please message #dcft-leonardo
</details>

<details>
<summary><h3>Where are the checkpoints saved?</h3></summary>

Checkpoints are saved at `$CHECKPOINTS_DIR`, which is `/leonardo_work/EUHPC_E03_068/DCFT_shared/checkpoints`. The final model will be saved here.


</details>

<details>
<summary><h3>Common error: FileNotFound</h3></summary>

If you see
```
FileNotFoundError: [Errno 2] No such file or directory: '/leonardo_work/EUHPC_E03_068/DCFT_shared/hub/mlfoundations-dev___s1_k-with-deepseek-r1-sharegpt/default/0.0.0/18f7e331953c21282965580e9a3637ed5b11eb41/cache-5928b541e9b720a7_00008_of_00016.arrow'
```
(From Marianna) This is because you need to pre-tokenize. Just run the same script (you can run it even on one node) with the tokenized_path set in your yaml config, it will tokenize and save everything. Then you can re-start and train. It is specific to Leonardo (very weird filesystem), on other machines I have never seen it, but it can be solved with pre-tokenization

I'm hoping that no one ever needs to deal with these details, but if you do, hopefully you save some headache and time. If anything messes up on the dataset loading side, try deleting the relevant directories and lock files in `$HF_HUB_CACHE`. 


(as of 2025.04.26) To pre-empt this issue if it happens frequently, first run the train script with `--pretokenize`. Currently, this only pretokenizes and does not launch training. The pretokenization step is fast, so you can quickly get through the slurm queue by speciying the `boost_qos_dbg` QOS, with a single node and 30min time limit. For example,

```
python3 -m hpc.launch --train_config_path dcft/train/hp_settings/paper/reasoning_large.yaml --time_limit 00:30:00 --num_nodes 1 --dataset mlfoundations-dev/openthoughts2 --model_name_or_path Qwen/Qwen2.5-32B-Instruct --pretokenize --qos boost_qos_dbg
```

The results of pretokenisation are saved to `$TOKENIZED_DATASETS_DIR`, which is `$DCFT/tokenized_datasets`. Then, relaunch the script with the `--pretokenize` flag, which will fetch the pretokenized datasets

```
python3 -m hpc.launch --train_config_path dcft/train/hp_settings/paper/reasoning_large.yaml --time_limit 1-00:00:00 --num_nodes 128 --dataset mlfoundations-dev/openthoughts2 --model_name_or_path Qwen/Qwen2.5-32B-Instruct --pretokenize --qos boost_qos_bprod
```



</details>

<details>
<summary><h3>How do I upload a model? </h3></summary>

Before uploading, check that the `README.md` of the relevant model names the expected base model. The base model may appear as something like `models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`. This will cause an error if you try to upload - edit these instances to `Qwen/Qwen2.5-32B-Instruct` before uploading:

```
find $CHECKPOINTS_DIR -name "openthoughts2" -type d | xargs -I{} sed -i 's|/leonardo_work/EUHPC_E03_068/DCFT_shared/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd|Qwen/Qwen2.5-32B-Instruct|g' {}/README.md
```

Then you can upload, with the CLI

```
find $CHECKPOINTS_DIR -name "openthoughts2" -type d | xargs -I{} bash -c 'MODEL_NAME=$(basename {}) && echo "Uploading $MODEL_NAME" && huggingface-cli upload mlfoundations-dev/$MODEL_NAME {} --exclude="checkpoint*" --repo-type=model --commit-message="Upload model"'
```

This is automated for you in two ways: 
```bash
python3 hpc/upload.py --model_name openthoughts2
```
or if `source $DCFT_PRIVATE/hpc/scripts/common.sh` in your `~/.bashrc`:
```
upload openthoughts2
```


</details>


<details>
<summary><h3>Common error: Got async event</h3></summary>

This means the node / interconnect is broken. The solution is to add the node to the exclusion list. The node number comes after lrdn. For example, below, the node number is 3031. 

```
[lrdn3031:0]:lrdn3031:166577:167079 [0] transport/net_ib.cc:101 NCCL WARN NET/IB : mlx5_2:1 Got async event :
```

Unfortunately there's a bit of unavoidable whack-a-mole here. Add your node number to the file `/leonardo_work/EUHPC_E03_068/DCFT_shared/dcft_private/hpc/sbatch/leonardo_train.sbatch` and launch your job again.
</details>


<details>
<summary><h3>How do I automate uploading models after training completes? </h3></summary>

if you are training lots of models - >5 finishing a day and its a lot to keep track of run the upload command over and over - you can you use the following from `common.sh` which will check for COMPLETED jobs every hour and upload the model
```
start_auto_upload
```
(warning this runs a background process, which you can find and kill with `stop_auto_upload` or `ps -u` + `kill <pid>`. you can see the logs with `status_auto_upload`)
</details>


## Related PRs

 - https://github.com/mlfoundations/dcft_private/pull/285
 - https://github.com/mlfoundations/dcft_private/pull/289
 - https://github.com/mlfoundations/dcft_private/pull/295
 - https://github.com/mlfoundations/dcft_private/pull/301

### Additional setup
```
conda config --add envs_dirs $DCFT_PRIVATE/env # can do `conda activate dcft_private`
conda config --add envs_dirs $EVALCHEMY/env # can do `conda activate cpu-evalchemy, cu121-evalchemy`
echo "source $DCFT_PRIVATE/hpc/dotenv/leonardo.env" >> ~/.bashrc # can use all the env variables (e.g. `cd $DCFT_PRIVATE`)
echo "source $DCFT_PRIVATE/database/access.env" >> ~/.bashrc # can use all the database credentials
```
