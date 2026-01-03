## GROUP SHARED SETUP
https://docs.tacc.utexas.edu/tutorials/sharingprojectfiles/
```bash
chmod g+X $WORK/..
chmod g+X $WORK
cd $WORK
mkdir dcft
chgrp -R G-827553 dcft
chgrp G-827553 $WORK
chgrp G-827553 $WORK/..
chmod g+s dcft
newgrp G-827553
chmod g+rwX dcft
```

## EVERYONE NEEDS TO ADD THIS TO THEIR BASHRC
```bash
cat <<'EOF' >> ~/.bashrc
# NEEDED FOR DCFT SHARED SETUP
if [[ "$PRIMARY_GROUP_SET" != "true" ]]; then
  export PRIMARY_GROUP_SET=true
  exec newgrp G-827553
fi
umask 007
export DCFT=/work/10159/rmarten/vista/dcft
export DCFT_PRIVATE=$DCFT/dcft_private
source $DCFT_PRIVATE/hpc/dotenv/tacc.env
EOF
source ~/.bashrc
```

## Continuing setup
```
cd $DCFT
mkdir $SCRATCH/checkpoints
mkdir hub
mkdir tokenized_datasets
git clone git@github.com:mlfoundations/dcft_private.git
git clone git@github.com:mlfoundations/evalchemy.git
```

## Helpful to have
```

cat << 'EOF' > ~/evaluate_pipeline.sh
#!/bin/bash
# Usage: ./evaluate_pipeline.sh <experiment>
experiment=$1
python $EVALCHEMY/eval/distributed/launch_simple.py --tasks AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo,CodeForces --num_shards 16 --max-job-duration 4 --model_name "mlfoundations-dev/${experiment}"
EOF
chmod +x ~/evaluate_pipeline.sh
echo "alias goeval=~/evaluate_pipeline.sh" >> ~/.bashrc
source ~/.bashrc

cat << 'EOF' > ~/evaluate_full.sh
#!/bin/bash
# Usage: ./evaluate_full.sh <model name>
model=$1
python $EVALCHEMY/eval/distributed/launch_simple.py --tasks AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo,CodeForces,AIME25,HLE,LiveCodeBenchv5 --num_shards 16 --max-job-duration 4 --model_name "${model}"
EOF
chmod +x ~/evaluate_pipeline.sh
echo "alias fulleval=~/evaluate_full.sh" >> ~/.bashrc
source ~/.bashrc
```


# OLD DOCS BELOW - DON'T READ THIS IS OUT OF DATE

Instruction for Getting Setup and Running Training Jobs on TACC

## Access Instructions

First, make an account [here](https://accounts.tacc.utexas.edu/begin).

After your account is approved, ping George on email / Slack so that you can be added to the allocation (this might take some time because the admins need to add you to the allocation).

A general guide on the cluster can be found [here](https://docs.tacc.utexas.edu/hpc/vista/), and some common pitfalls can be found in [this doc](https://docs.google.com/document/d/1URcWe8mLQF8HMNre7vZwgk6TiRNxFxVBwK_HCcRXQdM/edit?usp=sharing).

## Running training

To run training, you need to use a SIF that was built for dcft specifically - you can find this SIF file on TACC at `/scratch/10635/penfever/dcft_training.sif`. If needed, the SIF file can be rebuilt via:

```bash
module load tacc-apptainer
singularity build --nv dcft_training.sif dcft/train/singularity/TACC.def
```

To run training via sbatch, you should use the following command:
```bash
CONFIG_FILE=<your desired config file>
sbatch dcft/train/tacc/llamafactory_sbatch_tacc.sh $CONFIG_FILE
```

This will use 32 nodes (= 32 GPUs) for training.

## Tips on config files

Below are some good practices for training on TACC.
- For Llama3 finetuning, you should use a per device batch size of 16 and gradient accumulation 1.
- For DeepSpeed, use `dcft/train/llamafactory/examples/deepspeed/ds_z3_config.json` (the offload version doesn't seem to work.)
- It is advised to predownload the base model to `$SCRATCH/hf_home`, so that it is not downloaded to each node separately.
- Logs and checkpoints on each config file should point to `/tmp/dcft_checkpoints`, otherwise they will be discarded at end of training. 
