# Language Model Evaluation Harness

## Notice to Users
(as of 6/15/23)
We have a revamp of the Evaluation Harness library internals staged on the [big-refactor](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor) branch! It is far along in progress, but before we start to move the `master` branch of the repository over to this new design with a new version release, we'd like to ensure that it's been tested by outside users and there are no glaring bugs.

We’d like your help to test it out! you can help by:
1. Trying out your current workloads on the big-refactor branch, and seeing if anything breaks or is counterintuitive,
2. Porting tasks supported in the previous version of the harness to the new YAML configuration format. Please check out our [task implementation guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/docs/new_task_guide.md) for more information.

If you choose to port a task not yet completed according to [our checklist](https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/lm_eval/tasks/README.md), then you can contribute it by opening a PR containing [Refactor] in the name with:
- A command of the form `python main.py --model hf --model_args ..... --tasks <task name> ...` which will run the task in the `master` branch, and what the score is
- A command of the form `python main.py --model hf --model_args ..... --tasks <task name> ...` to run the task in your PR branch to `big-refactor`, and what the resulting score is, to show that we achieve equality between the two implementations.

Lastly, we'll no longer be accepting new feature requests beyond those that are already open to the master branch as we carry out this switch to the new version over the next week, though we will be accepting bugfixes to `master` branch and PRs to `big-refactor`. Feel free to reach out in the #lm-thunderdome channel of the EAI discord for more information.


## Overview

This project provides a unified framework to test generative language models on a large number of different evaluation tasks.

Features:

- Many tasks implemented, 200+ tasks implemented in the old framework which require porting to the new setup as described in https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/lm_eval/docs/new_task_guide.md.
- Support for models loaded via [transformers](https://github.com/huggingface/transformers/) (including quantization via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)), [GPT-NeoX](https://github.com/EleutherAI/gpt-neox), and [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/), with a flexible tokenization-agnostic interface.
- Support for commercial APIs including [OpenAI](https://openai.com), [goose.ai](https://goose.ai), and [TextSynth](https://textsynth.com/).
- Support for evaluation on adapters (e.g. LoRa) supported in [HuggingFace's PEFT library](https://github.com/huggingface/peft).
- Evaluating with publicly available prompts ensures reproducibility and comparability between papers.

## Install

To install the `lm-eval` refactor branch from the github repository, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
git checkout big-refactor
pip install -e .
```

To install additional multilingual tokenization and text segmentation packages, you must install the package with the `multilingual` extra:

```bash
pip install -e ".[multilingual]"
```

To support loading GPTQ quantized models, install the package with the `gptq` extra:

```bash
pip install -e ".[gptq]"
```

## Basic Usage

### Hugging Face `transformers`

To evaluate a model hosted on the [HuggingFace Hub](https://huggingface.co/models) (e.g. GPT-J-6B) on `hellaswag` you can use the following command:


```bash
python main.py \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

Additional arguments can be provided to the model constructor using the `--model_args` flag. Most notably, this supports the common practice of using the `revisions` feature on the Hub to store partially trained checkpoints, or to specify the datatype for running a model:

```bash
python main.py \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks lambada_openai,hellaswag \
    --device cuda:0 \
    --batch_size 8
```

Models that are loaded via either `transformers.AutoModelForCausalLM` (autoregressive, decoder-only GPT style models) or `transformers.AutoModelForSeq2SeqLM` (such as encoder-decoder models like T5) in Huggingface are supported via  Support for this model type is currently pending.

### Multi-GPU Evaluation with Hugging Face `accelerate`

To parallelize evaluation of HuggingFace models across multiple GPUs, we allow for two different types of multi-GPU evaluation.

The first is performed by launching evaluation via the `accelerate` library as follows:

```
accelerate launch main.py \
    --model hf \
    --tasks lambada_openai,arc_easy \
    --batch_size 16 \
```

This will perform *data-parallel evaluation*: that is, placing a **single full copy** of your model onto each available GPU and *splitting batches across GPUs* to evaluate on K GPUs K times faster than on one.

However, if your model *is too large to be run on a single one of your GPUs*, then we provide an alternative method to run these large models: use of the `parallelize` argument.

```
python main.py \
    --model hf \
    --model_args pretrained=EleutherAI/pythia-12b,parallelize=True
    --tasks lambada_openai,arc_easy \
    --batch_size 16
```

To pass even more advanced keyword arguments to `accelerate`, we allow for the following arguments as well:
- `device_map_option`: How to split model weights across available GPUs. defaults to "auto".
- `max_memory_per_gpu`: the max GPU memory to use per GPU in loading the model.
- `max_cpu_memory`: the max amount of CPU memory to use when offloading the model weights to RAM.
- `offload_folder`: a folder where model weights will be offloaded to disk if needed.

Using this setting helps for massive models like BLOOM which require, or to avoid exceeding your total system RAM (by default, with `accelerate launch` one copy of the model for each GPU is initialized in RAM before moving it to GPU, resulting in large RAM usage spikes around the start of the script that may cause errors such as `Killed`.) However, it naively splits models across GPUs, resulting in only a single GPU performing work at any point in time, and so is much slower than launching with `accelerate launch`, possibly by a factor of the total # of GPUs.

**Note that this option requires launching evaluation via `python main.py` rather than `accelerate launch main.py`.**

### Commercial APIs

Our library also supports language models served via the OpenAI API:

```bash
export OPENAI_API_SECRET_KEY=YOUR_KEY_HERE
python main.py \
    --model openai \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag
```

While this functionality is only officially maintained for the official OpenAI API, it tends to also work for other hosting services that use the same API such as [goose.ai](goose.ai) with minor modification. We also have an implementation for the [TextSynth](https://textsynth.com/index.html) API, using `--model textsynth`.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
python main.py \
    --model openai \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

### Other Frameworks

A number of other libraries contain scripts for calling the eval harness through their library. These include [GPT-NeoX](https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py), [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/blob/main/examples/MoE/readme_evalharness.md), and [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py).

### Additional Features

If you have a CUDA-compatible Mac GPU, you can run the eval harness using the MPS back-end by replaicng `--device cuda:0` with `--device mps:0`. PyTorch does not currently support automatic mixed precision (AMP) for MPS, so we forcibly cast all weights to fp32 regardless of how they're stored. This is slower and has a larger memory footprint than we can achieve on Linux systems, but as PyTorch continues to improve its MPS support we hope to continue to improve it.

💡 **Tip**: You can inspect what the LM inputs look like by running the following command:

```bash
python write_out.py \
    --tasks all_tasks \
    --num_fewshot 5 \
    --num_examples 10 \
    --output_base_path /path/to/output/folder
```

This will write out one text file for each task.

## Advanced Usage

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:
```bash
python main.py \
    --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

[GPTQ](https://github.com/PanQiWei/AutoGPTQ) quantized models can be loaded by specifying their file names in `,gptq=NAME` (or `,gptq=True` for default names) in the `model_args` argument:

```bash
python main.py \
    --model hf \
    --model_args pretrained=model-name-or-path,gptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

## Implementing new tasks

To implement a new task in the eval harness, see [this guide](./docs/new_task_guide.md).


As a start, we currently only support one prompt per task, which we strive to make the "standard" as defined by the benchmark's authors. If you would like to study how varying prompts causes changes in the evaluation score, we support prompts authored in the [Promptsource Library](https://github.com/bigscience-workshop/promptsource/tree/main) as described further in https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/lm_eval/docs/new_task_guide.md and https://github.com/EleutherAI/lm-evaluation-harness/blob/big-refactor/lm_eval/docs/advanced_task_guide.md and welcome contributions of novel task templates and task variants.

## Cite as

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
