
# Instruction for Getting Setup and Running Training Jobs on ZIH

## Get a ZIH Account

Fill out this application form:

[ZIH Account Application](https://selfservice.zih.tu-dresden.de/index.php/hpclogin/noLogin)

with the following details:

- **Name der Einrichtung**: [name of your institution]
- **Project title**: Data Competition Fine-tuning
- **Project expiry**: 17.09.2025
- **Project manager**: Reinhard Heckel
- Check the box: *Ich akzeptiere die Nutzungsbedingungen für HPC* (I accept the HPC user guidelines)

Send the form to **Reinhard Heckel** for signing. He will sign and forward the form to `servicedesk@tu-dresden.de`.

---

## Setup on ZIH

Once you have an account, you can log in as follows:

### VPN Connection
You need to be connected to the VPN:
[ZIH VPN Access](https://tu-dresden.de/zih/dienste/service-katalog/arbeitsumgebung/zugang_datennetz/vpn)

### Cluster Description
[ZIH HPC Cluster Information](https://tu-dresden.de/zih/hochleistungsrechnen/hpc)

### Login Information
- **For training jobs us the cluster with nodes with 8xA100 40GB**: 
  ```sh
  ssh your_username@login2.alpha.hpc.tu-dresden.de
  ```
- **For data generation, filtering, etc. (without GPUs)**:
  ```sh
  ssh your_username@login1.barnard.hpc.tu-dresden.de
  ```

---

## Shared Setup (new)

### How to use the shared setup
```sh
echo "source /data/cat/ws/ryma833h-dcft/dcft_private/hpc/dotenv/zih.env" >> ~/.bashrc
```

Then run the training command (same on all HPC clusters)
```sh
python3 -m hpc.launch \
    --dataset=mlfoundations-dev/OpenThoughts2-1M-scrubbed \
    --num_nodes=32 \
    --time_limit=00:10:00 \
    --config_path=dcft/train/hp_settings/reasoning.yaml
```

### How the shared setup was created (already done for you)
```sh
# Allocate a group writable workspace
# Guide: https://compendium.hpc.tu-dresden.de/data_lifecycle/workspaces/#grant-write-permissions-for-a-group
groups # you should see p_finetuning
ws_allocate --groupname p_finetuning --filesystem cat --reminder 7 --mailaddress ryan.marten@mailbox.tu-dresden.de --name dcft --duration 30

# Define all project paths
cat << 'EOF' > ~/zih.env
export DCFT=/data/cat/ws/ryma833h-dcft
export DCFT_GROUP=p_finetuning
export DCFT_CONDA=$DCFT/miniconda3
export DCFT_PRIVATE=$DCFT/dcft_private
export DCFT_PRIVATE_ENV=$DCFT_PRIVATE/env/dcft_private
export DCFT_PRIVATE_ACTIVATE_ENV="source $DCFT_CONDA/bin/activate $DCFT_PRIVATE_ENV"
export EVALCHEMY=$DCFT/evalchemy
export EVALCHEMY_ENV=$EVALCHEMY/env/evalchemy
export EVALCHEMY_ACTIVATE_ENV="source $DCFT_CONDA/bin/activate $EVALCHEMY_ENV"
EOF
source ~/zih.env
mkdir -p $HF_HUB_CACHE

# Clone repos
git clone git@github.com:mlfoundations/evalchemy.git $EVALCHEMY
git clone git@github.com:mlfoundations/dcft_private.git $DCFT_PRIVATE

# Set up conda in the shared workspace
mkdir -p $DCFT_CONDA
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $DCFT_CONDA/miniconda.sh
bash $DCFT_CONDA/miniconda.sh -b -u -p $DCFT_CONDA
rm $DCFT_CONDA/miniconda.sh
source $DCFT_CONDA/bin/activate

# Create conda environment for evalchemy
cd $EVALCHEMY
conda create -y --prefix $EVALCHEMY_ENV python=3.10
conda activate $EVALCHEMY_ENV
pip install -e .
pip install -e eval/chat_benchmarks/alpaca_eval
conda deactivate

# Create conda environment for dcft_private
cd $DCFT_PRIVATE
conda create -y --prefix $DCFT_PRIVATE_ENV python=3.10
conda activate $DCFT_PRIVATE_ENV
pip install -r dcft/train/requirements.txt
pip install -e .
conda deactivate

# For others conveinence
mv ~/zih.env $DCFT_PRIVATE/hpc/dotenv/zih.env

# For your conveinence
conda init # conda is added to your shell
conda config --add envs_dirs $DCFT_PRIVATE/env # can do `conda activate dcft_private`
conda config --add envs_dirs $EVALCHEMY/env # can do `conda activate evalchemy`
echo "source $DCFT_PRIVATE/hpc/dotenv/zih.env" >> ~/.bashrc # can use all the env variables (e.g. `cd $DCFT_PRIVATE`)
echo "source $DCFT_PRIVATE/hpc/scripts/common.sh" >> ~/.bashrc # database credentials and common commands
```

## Individual Setup (old)
### Setup Miniconda and Environment

```sh
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

conda create --name dcfttrain python=3.10
conda activate dcfttrain
```

### Clone the Repository and Install Dependencies

```sh
git clone git@github.com:mlfoundations/dcft_private.git
cd dcft_private/dcft/train
pip install -r requirements.txt
```

---

### Setup a Workspace for Data and Checkpoints

This gets you a WORK folder, details are here [Workspace Allocation Guide](https://doc.zih.tu-dresden.de/quickstart/getting_started/?h=workspaces#allocate-a-workspace):

```sh
ws_allocate -F horse -r 7 -m marie@tu-dresden.de -n number_crunch -d 90
export WORK=/data/horse/ws/marie-number_crunch
```


### Set Environment Variables

Store environment variables in `dcft_private/.env`:

```sh
cat <<EOF > dcft_private/.env
WORK=/data/horse/ws/marie-number_crunch
HF_TOKEN=
DB_PASSWORD=
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_PROJECT=dcft
OPENAI_API_KEY=
EOF
```

To load the environment variables, use the below but the train script does that for you:

```sh
export $(grep -v '^#' dcft_private/.env | xargs)
```

---

## Schedule Training Jobs

To submit a training job:

```sh
sbatch dcft_private/dcft/train/zih/llamafactory_sbatch_zih.sh dcft_private/dcft/train/configs/mammoth/llama3_mammoth_dcft_ablation_original_50k.yaml
```

To get an interactive node:

```sh
salloc --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --time=04:00:00 --mem=768G --exclusive
```

## Notes
Original pull request in which this was added (https://github.com/mlfoundations/dcft_private/pull/283)


## Individual setup 
You can create your own clones to do development by following the simliar instructions above, but creating a new workspace and then changing any relevant paths. 
