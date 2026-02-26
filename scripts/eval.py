import argparse
from typing import List

from src.eval_tasks import EVAL_TASKS, get_task
from src.evaluator import add_common_args, run_eval


def _infer_model_tag(model_id: str) -> str:
    model_name = model_id.split("/")[-1]
    model_tag = model_name.split("-")[0] if "-" in model_name else model_name
    return model_tag.lower()


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(EVAL_TASKS.keys()))
    add_common_args(parser)

    # Sweep options (optional)
    parser.add_argument("--model_seeds", default=None, help="Comma-separated list, e.g. 42,99,123")
    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument("--ks", default=None, help="Comma-separated hot-k list, e.g. 4,8,12,16")
    sweep_group.add_argument(
        "--coverages",
        default=None,
        help="Comma-separated coverage tags for cov adapters, e.g. 60,70,60p5",
    )
    parser.add_argument(
        "--adapter_template",
        default=None,
        help=(
            "Template placeholders: {seed} plus either {k} (hot) or {coverage}/{cov} (dyn). "
            "Examples: AManzoni/olmoe_gsm8k_s{seed}_hotk{k} | "
            "AManzoni/olmoe_spider_s{seed}_cov{coverage}"
        ),
    )
    parser.add_argument(
        "--run_name_template",
        default=None,
        help=(
            "Optional template supporting {seed},{task},{model_tag},{k},{coverage},{cov}. "
            "Example: {model_tag}_{task}_s{seed}_cov{coverage}"
        ),
    )
    parser.add_argument("--dry_run", action="store_true")

    args, _ = parser.parse_known_args()
    task = get_task(args.task)
    if hasattr(task, "add_args") and task.add_args:
        task.add_args(parser)

    args = parser.parse_args()

    sweep_requested = bool(args.adapter_template or args.model_seeds or args.ks or args.coverages)
    if sweep_requested:
        if not (args.adapter_template and args.model_seeds and (args.ks or args.coverages)):
            raise ValueError("Sweep requires --adapter_template, --model_seeds, and one of --ks/--coverages")

        model_tag = _infer_model_tag(args.model)
        model_seeds = _parse_int_list(args.model_seeds)
        if args.ks:
            sweep_values = [str(k) for k in _parse_int_list(args.ks)]
            sweep_kind = "k"
        else:
            sweep_values = _parse_str_list(args.coverages)
            sweep_kind = "coverage"
            if not sweep_values:
                raise ValueError("--coverages must contain at least one value")

        for seed in model_seeds:
            for value in sweep_values:
                adapter = args.adapter_template.format(
                    seed=seed,
                    k=value,
                    coverage=value,
                    cov=value,
                    task=args.task,
                    model_tag=model_tag,
                )

                if args.run_name_template:
                    run_name = args.run_name_template.format(
                        model_tag=model_tag,
                        task=args.task,
                        seed=seed,
                        k=value,
                        coverage=value,
                        cov=value,
                    )
                else:
                    run_name = None

                if args.dry_run:
                    print(
                        f"[DRY RUN] task={args.task} sweep={sweep_kind}:{value} "
                        f"adapter={adapter} run_name={run_name or '<auto>'}"
                    )
                    continue

                run_args = argparse.Namespace(**vars(args))
                run_args.adapter = adapter
                run_args.run_name = run_name
                run_eval(args.task, run_args)
        return

    run_eval(args.task, args)


if __name__ == "__main__":
    main()
