<div style="text-align: center;">

  <h1>
    <a href="https://gpt-nl.nl/" target="_blank"><img src="https://gpt-nl.nl/publish/pages/5180/beeldmerk-gpt_nl.svg" alt="GPT-NL" width="200"></a>
    <br>
    GPT-NL Pre-training scripts
  </h1>

#### Training and evaluation, supporting the construction of the <a href="https://gpt-nl.nl" target="_blank">GPT-NL</a> model.

<a href="#installation">Installation</a> &nbsp;&middot;&nbsp;
<a href="#usage">Usage</a> &nbsp;&middot;&nbsp;
<a href="#development-notes">Development notes</a>

</div>

> This repository is part of [GPT-NL](https://gpt-nl.nl/), a responsible alternative to existing language models, using lawfully obtained Dutch data and sharing proceeds with copyright holders to promote fairness and strengthen Europe's digital position.
> We aim to be as transparent as possible about how the model has been created. This open source repository should be considered as an effort to support this transparency and fairness, not to be seen as a working product of any kind.  
> For any questions regarding GPT-NL, check our [website](https://gpt-nl.nl/). Any questions regarding this specific repository can be sent to [Anne Stijns](mailto:anne.stijns@tno.nl) or submitted as an issue in the issue tracker of this repository.

This repository contains scripts used to train GPT-NL on a SLURM runner and evaluate its performance.
Performance is evaluated using [EuroEval](https://euroeval.com/), among others.

## Installation

To install the repository, you need to:

1. Make sure you are cloning via ssh and that you don't have a passphrase on your ssh key (otherwise pulling inside the production job will fail)
2. `./scripts/installation/install_olmo_snellius.sh`: creates a local environment at `./venv`

### Set up build directories

To train GPT-NL, various directories need to be setup to store tokenized from [the GPT-NL data loader](https://github.com/GPT-NL/dataloader), checkpoints during training, et cetera.
This open-source repository uses some placeholders for these directories:

|Placeholder text| Directory purpose | Example |
|---|---|---|
|INSERT-GPT-NL-DATALOADER-DIR| Tokenized inputs from the GPT-NL data loader | `/GPT-NL/dataloader/`
|INSERT-GPT-NL-TRAINING-DIR| Checkpoints, debug datasets, and context lengethening weights | `/GPT-NL/training/`
|INSERT-GPT-NL-ROOT-DIR| Scratch space during training | `/GPT-NL/`

The placeholders can be replaced with actual values using a text replacement tool such as [sed](https://linux.die.net/man/1/sed).
Edit `project-placeholders.sed` to fit your configuration, then run

```bash
rg '/INSERT-GPT-NL-[[:alpha:]]+-DIR/' -c | cut -d : -f 1 | xargs sed -i -f project-placeholders.sed
```

###  Setting up experiments

Create `./scripts/snellius/.env` with the following content

```bash
export WANDB_ENTITY=<INSERT-USER-HERE>
export WANDB_API_KEY=<INSERT-KEY-HERE> # get from https://wandb.ai/authorize
export HF_KEY=<INSERT-KEY_HERE> # get from https://huggingface.co/settings/tokens to run euroeval for HF models online
```

<!--
Consider:

- You can assume a basic level of understanding. Not _every_ step has to be explained extensively - often listing the commands is sufficient. Example: you don't have to explain the concept of virtual environments.
- Be complete and assume that people are starting fresh. Example: don't assume people have a specific conda or a specific Python version.
- People use different operating systems.
-->

## Usage

This repository has two slurm jobs to start GPT-NL training

```
# for production (real training, automatically pulls and switches to main)
sbatch scripts/snellius/launch_train_prod.job --auto-restart 1
# for debug (no notifications and no main required, to test out things)
sbatch scripts/snellius/launch_train_debug.job


Arguments:
   --profiling   run NVIDIA Nsight Systems (nsys) profiler. Adjust stop_at to 7 steps
   --gpu-health-check   perform GPU health check on all requested nodes
   --auto-restart <int>    upon 60 seconds before walltime expiration, re-submit the job from the latest checkpoint <int> number of times

See scripts/snellius/launch_train_prod.job for more information
```

### EuroEval

For running EuroEval, you need first to create a separate python environment with `bash scripts/evaluation/install_eval.sh` which will create an environment in the folder `./venv_eval`. Conflicts between EuroEval and flash-attn make this necessary.

To run [EuroEval](https://github.com/EuroEval/EuroEval/) on a checkpoint:
1. run conversion + evaluation with `sbatch scripts/evaluation/run_euroeval.sh <CHECKPOINT_PATH>`:
  a. if you pass a OLMO-core checkpoint path, it will be converted first
  b. if you pass a checkpoint already converted at `/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints`, only evaluation will be done


This will convert the checkpoint to HF format (if not already in folder), run the Euroeval benchmark and write the results to Wandb (and to json file in checkpoint dir) in: `https://wandb.ai/<INSERT-USER-HERE>/euroeval/workspace`

## Development notes

### Updating base OLMo-core code
GPT-NL uses a fork of OLMo-core with patches specific to the GPT-NL model.
This fork is available [here](https://github.com/GPT-NL/OLMo-core).
Making changes to the code based on the original OLMo-core ai2-olmo-core package is done through the fork.

Then make sure you reinstall the updated `olmo-core` dependency by using `scripts/installation/update_olmo_local_environment.sh`.

### Inference

In the [scripts/inference](./scripts/inference/) folder you can find the conversion script to load the model with huggingface.
