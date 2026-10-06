import os
import re
from argparse import ArgumentParser

import wandb
from euroeval import Benchmarker
from olmo_core.utils import prepare_cli_environment


def extract_run_and_step(path):
    """ "
    Extract run and step from path, assuming HF path is formatting like path/to/run/step
    """
    parts = path.strip("/").split("/")
    run_name = parts[-2]
    step_str = parts[-1]
    # step_str = parts[-1]
    print("step_str", step_str)
    step_num = int(re.match(r"step(\d+)", step_str).group(1))
    # run_name = f"GPTNL-26B-step-{step_num}"
    run_name = f"{run_name}/{step_str}"
    return run_name, step_num


def set_cwd(model_name: str):
    if not os.path.isdir(model_name):
        # remote models don't have a corresponding dir yet
        print(f"creating directory {model_name}")
        os.makedirs(model_name)
    os.chdir(model_name)


def evaluate(args):
    set_cwd(args.model)  # we cannot set output dir for results, so changing working dir to specific checkpoint
    # NB: if output json already exists, this leads to hanging/skipping of evaluation
    benchmark = Benchmarker()
    languages = list(set(args.language))  # remove duplicates
    benchmark.benchmark(
        model=args.model, language=languages, task=args.task, api_key=args.api_key, cache_dir=args.cache_dir
    )
    results = benchmark.benchmark_results
    print("results", results)

    print(f"args.model={args.model} args.run_name={args.run_name} and args.step={args.step}")
    if args.run_name and args.step != None:
        run_name = args.run_name
        step_num = args.step
        # OLMo-2-32B has version from branch name (@branch)
        group = args.run_name.split("/")[-1].split("@")[-2]
    else:
        run_name, step_num = extract_run_and_step(args.model)
        group = "GPTNL-26B-202506"
    wandb.init(project="euroeval", name=run_name, resume="allow", group=group)
    for benchmark_result in results:
        prefix = benchmark_result.dataset  # e.g., 'dbrd'

        # Log all keys in each raw result
        for i, r in enumerate(benchmark_result.results["raw"]):
            for key, value in r.items():
                wandb.log({f"{prefix}/{key}_{i}": value}, step=step_num)

        # Log all aggregated metrics in 'total'
        for key, value in benchmark_result.results["total"].items():
            wandb.log({f"{prefix}/{key}": value}, step=step_num)

    wandb.finish()


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Local or remote directory containing the HF checkpoint, or the model id of a HF Hub repo.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=["nl"],
        action="append",
        help="Languages on which to evaluate the model",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task to evaluate on. If none, all tasks will be benchmarked. Example task: sentiment-classification",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for logging on wandb. If None, will take from model path",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step for logging on wandb. If None, will take from model path",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for remote models",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache dir to use for remote models",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    evaluate(args)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
