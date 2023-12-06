import argparse
import numpy as np
import lm_eval.evaluator
from lm_eval import tasks
import scipy.stats
from typing import Tuple, Dict

eval_logger = lm_eval.utils.eval_logger


def calculate_z_value(res1: Dict, res2: Dict, limit: int) -> Tuple[float, float]:
    acc1, acc2 = res1["acc,none"], res2["acc,none"]
    st_err1, st_err2 = res1["acc_stderr"], res2["acc_stderr"]
    Z = (acc1 - acc2) / np.sqrt((st_err1**2 / limit) + (st_err2**2 / limit))
    # Determining the p-value
    p_value = 2 * scipy.stats.norm.sf(abs(Z))  # two-tailed test
    return Z, p_value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained", default="EleutherAI/pythia-70m", help="name of model to compare"
    )
    parser.add_argument(
        "--hf_args", help="huggingface model args <arg>=<value>", default=""
    )
    parser.add_argument("--vllm_args", help="vllm model args <arg>=<value>", default="")
    parser.add_argument("--tasks", type=str, default="arc_easy,hellaswag")
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Logging verbosity",
    )
    return parser.parse_args()


if __name__ == "__main__":
    tasks.initialize_tasks()
    args = parse_args()
    tasks = args.tasks.split(",")
    print(tasks)
    hf_args = "," + args.hf_args
    vllm_args = "," + args.vllm_args
    results_hf = lm_eval.evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.pretrained}" + hf_args,
        tasks=tasks,
        limit=args.samples,
        device=args.device,
        batch_size=args.batch,
    )
    results_vllm = lm_eval.evaluator.simple_evaluate(
        model="vllm",
        model_args=f"pretrained={args.pretrained}" + vllm_args,
        tasks=tasks,
        limit=args.samples,
        device=args.device,
        batch_size=args.batch,
    )
    all_res = {}
    for task1, task2 in zip(
        results_hf["results"].items(), results_vllm["results"].items()
    ):
        assert task1[0] == task2[0]
        z, p_value = calculate_z_value(task1[1], task2[1], args.samples)
        all_res[task1[0]] = {"z": z, "p_value": p_value}
        assert p_value > 0.05
        eval_logger.info(all_res)
