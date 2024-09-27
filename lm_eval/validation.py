import json
import logging
import os
import sys

from lm_eval import evaluator, utils
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.loggers import EvaluationTracker, WandbLogger
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string


from dotenv import load_dotenv, find_dotenv

# Load the environment variables from the .env file
load_dotenv(find_dotenv('.env'))

def validate_params(params: dict) -> None:
    # Set up logging
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, params['verbosity']))
    eval_logger.info(f"Verbosity set to {params['verbosity']}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set up WandbLogger if needed
    if params['wandb_args']:
        wandb_logger = WandbLogger(**simple_parse_args_string(params['wandb_args']))

    # Update evaluation tracker args
    if params['output_path']:
        params['hf_hub_log_args'] += f",output_path={params['output_path']}"
    if os.environ.get("HF_TOKEN", None):
        params['hf_hub_log_args'] += f",token={os.environ.get('HF_TOKEN')}"
    evaluation_tracker_args = simple_parse_args_string(params['hf_hub_log_args'])
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if params['predict_only']:
        params['log_samples'] = True
    if (params['log_samples'] or params['predict_only']) and not params['output_path']:
        raise ValueError(
            "Specify --output_path if providing --log_samples or --predict_only"
        )

    if params['fewshot_as_multiturn'] and params['apply_chat_template'] is False:
        raise ValueError(
            "When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set (either to `True` or to the chosen template name)."
        )

    task_manager = TaskManager(params['verbosity'], include_path=params['include_path'])

    if "push_samples_to_hub" in evaluation_tracker_args and not params['log_samples']:
        eval_logger.warning(
            "Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub."
        )

    if params['limit']:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    # Handle tasks
    if params['tasks'] is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    else:
        task_list = params['tasks'].split(",")
        task_names = task_manager.match_tasks(task_list)
        for task in [task for task in task_list if task not in task_names]:
            if os.path.isfile(task):
                config = utils.load_yaml_config(task)
                task_names.append(config)
        task_missing = [
            task for task in task_list if task not in task_names and "*" not in task
        ]

        if task_missing:
            missing = ", ".join(task_missing)
            eval_logger.error(
                f"Tasks were not found: {missing}\n"
                f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
            )
            raise ValueError(
                f"Tasks not found: {missing}. Try using valid task names."
            )

    # Handle trust_remote_code
    if params['trust_remote_code']:
        eval_logger.info(
            "Passed `trust_remote_code=True`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        import datasets
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        params['model_args'] = params['model_args'] + ",trust_remote_code=True"

    eval_logger.info(f"Selected Tasks: {task_names}")

    request_caching_args = request_caching_arg_to_dict(
        cache_requests=params['cache_requests']
    )

    results = evaluator.simple_evaluate(
        model=params['model'],
        model_args=params['model_args'],
        tasks=task_names,
        num_fewshot=params['num_fewshot'],
        batch_size=params['batch_size'],
        max_batch_size=params['max_batch_size'],
        device=params['device'],
        use_cache=params['use_cache'],
        limit=params['limit'],
        check_integrity=params['check_integrity'],
        write_out=params['write_out'],
        log_samples=params['log_samples'],
        evaluation_tracker=evaluation_tracker,
        system_instruction=params['system_instruction'],
        apply_chat_template=params['apply_chat_template'],
        fewshot_as_multiturn=params['fewshot_as_multiturn'],
        gen_kwargs=params['gen_kwargs'],
        task_manager=task_manager,
        verbosity=params['verbosity'],
        predict_only=params['predict_only'],
        random_seed=params['seed'][0],
        numpy_random_seed=params['seed'][1],
        torch_random_seed=params['seed'][2],
        fewshot_random_seed=params['seed'][3],
        **request_caching_args,
    )

    if results is not None:
        if params['log_samples']:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if params['show_config']:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # Add W&B logging
        if params['wandb_args']:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if params['log_samples']:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:
                eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if params['log_samples'] else None
        )

        if params['log_samples']:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if (
            evaluation_tracker.push_results_to_hub
            or evaluation_tracker.push_samples_to_hub
        ):
            evaluation_tracker.recreate_metadata_card()

        print(
            f"{params['model']} ({params['model_args']}), gen_kwargs: ({params['gen_kwargs']}), limit: {params['limit']}, num_fewshot: {params['num_fewshot']}, "
            f"batch_size: {params['batch_size']}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        processed_results = make_table(results)
        group_results = make_table(results, "groups") if "groups" in results else None

        if params['wandb_args']:
            # Tear down wandb run once all the logging is done.
            wandb_logger.run.finish()

        return processed_results, group_results
        
        # return results,group_results 

        # print(make_table(results))
        # if "groups" in results:
        #     print(make_table(results, "groups"))

        # if params['wandb_args']:
        #     # Tear down wandb run once all the logging is done.
        #     wandb_logger.run.finish()


# if __name__ == "__main__":
    
#     model_name = "davinci-002"
#     # tasks = "mmlu,hellaswag,anli,glue,bigbench_multiple_choice"
#     # tasks = "anli,arc_challenge,arithmetic,asdiv,bigbench_multiple_choice,blimp,commonsense_qa,coqa,drop,eq_bench,fda,glue,gpqa,gsm8k,hellaswag,inverse_scaling_mc,lambada,leaderboard,mathqa,med_concepts_qa,mmlu,mmlusr,mutual,qasper,squadv2,super-glue-lm-eval-v1,truthfulqa,unscramble,wikitext"
#     tasks = "super-glue-lm-eval-v1"
    
#     output_path = "D:\\Code\\Python\\office\\LLM-Evaluation-Framework-Using-LM-Harness\\results\\"+model_name+"_"+tasks.replace(",","_")+"\\"

#     params = {
#         "model": "openai-completions",
#         "model_args": "model="+model_name,
#         "tasks": tasks,
#         "num_fewshot": None,
#         "batch_size": 1,
#         "max_batch_size": None,
#         "device": None,
#         "output_path": None,
#         "limit": 0.01,
#         "use_cache": None,
#         "cache_requests": True,
#         "check_integrity": False,
#         "write_out": False,
#         "log_samples": True,
#         "system_instruction": None,
#         "apply_chat_template": False,
#         "fewshot_as_multiturn": False,
#         "show_config": False,
#         "include_path": None,
#         "gen_kwargs": None,
#         "verbosity": "INFO",
#         "wandb_args": "",
#         "hf_hub_log_args": "hub_results_org=SameedHussain,push_results_to_hub=True,push_samples_to_hub=True,token=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,public_repo=True",
#         "predict_only": False,
#         "seed": [0, 1234, 1234, 1234],
#         "trust_remote_code": True,
#     }

#     validate_params(params)
