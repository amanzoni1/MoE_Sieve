import argparse
import os
from huggingface_hub import HfApi
from src.utils_profiling import build_hotmap, build_hotmap_by_coverage
from src.trainer import run_training
from src.data_registry import DATASETS
from src.config import TRAIN_CFG


def _coverage_tag(coverage_pct: float) -> str:
    return f"{coverage_pct:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description="Experiment Launcher")

    # --- Essentials ---
    parser.add_argument("--task", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--mode", type=str, default="hot", choices=["hot", "dyn", "full", "random"])
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help=f"Override base model id (default from config: {TRAIN_CFG.model_id})",
    )
    parser.add_argument("--model_tag", type=str, default=None, help="Short model tag for names (e.g. olmoe)")
    parser.add_argument(
        "--run_name_suffix",
        type=str,
        default="",
        help="Optional suffix appended to auto run_name (example: _3e).",
    )
    parser.add_argument("--seed", type=int, default=None, help=f"Override ({TRAIN_CFG.seed})")
    parser.add_argument(
        "--data_seed",
        type=int,
        default=None,
        help="Dataset shuffle seed. Defaults to --seed when not set.",
    )

    # --- Hot Mode ---
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "-cov",
        "--coverage_pct",
        type=float,
        default=None,
        help="Coverage target percent for mode=dyn (example: -cov 60).",
    )
    parser.add_argument("--random_seed", type=int, default=None, help="Random expert selector seed (mode=random)")
    parser.add_argument("--telemetry", type=str, help="Path to telemetry .pt")
    parser.add_argument("--hotmap", type=str, help="Path to hotmap .json")
    parser.add_argument("--hotmap_dir", type=str, default=None, help="Optional dir for hotmap_template")
    parser.add_argument(
        "--hotmap_template",
        type=str,
        default=None,
        help="Filename template with {k}, {coverage_pct}, {task}, {seed}, {mode}.",
    )
    parser.add_argument("--hotmap_mode", type=str, default="counts", choices=["counts", "mass"])
    parser.add_argument(
        "--coverage_min_k",
        type=int,
        default=None,
        help="Optional lower clamp for per-layer k in coverage mode.",
    )
    parser.add_argument(
        "--coverage_max_k",
        type=int,
        default=None,
        help="Optional upper clamp for per-layer k in coverage mode.",
    )

    # --- Training Overrides (Default None = Use Registry/Config) ---
    parser.add_argument("--lr", type=float, default=None, help="Override Registry LR")
    parser.add_argument("--epochs", type=int, default=None, help="Override Registry Epochs")
    parser.add_argument("--train_samples", type=int, default=None, help="Debug: limit train samples")
    parser.add_argument("--bs", type=int, default=None, help=f"Override Config BS ({TRAIN_CFG.per_device_bs})")
    parser.add_argument("--grad_acc", type=int, default=None, help=f"Override Config GradAcc ({TRAIN_CFG.grad_acc})")
    parser.add_argument("--max_len", type=int, default=None, help=f"Override ({TRAIN_CFG.max_len})")
    parser.add_argument("--r", type=int, default=None, help=f"Override ({TRAIN_CFG.r})")
    parser.add_argument("--alpha", type=int, default=None, help=f"Override ({TRAIN_CFG.alpha})")
    parser.add_argument("--dropout", type=float, default=None, help=f"Override ({TRAIN_CFG.dropout})")

    # --- Tracking (W&B) ---
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    # --- Hub ---
    parser.add_argument("--no_push_to_hub", action="store_false", dest="push_to_hub")
    parser.add_argument("--hub_repo", type=str, default=None, help="Repo name")
    parser.add_argument("--hub_public", action="store_false", dest="hub_private")

    # --- Cleanup ---
    parser.add_argument(
        "--cleanup_after_push",
        action="store_true",
        help="Delete local run outputs after successful Hub push",
    )

    args = parser.parse_args()
    seed_eff = args.seed if args.seed is not None else TRAIN_CFG.seed
    model_id_eff = args.model_id if args.model_id else TRAIN_CFG.model_id

    if args.mode == "dyn":
        if args.coverage_pct is None:
            raise ValueError("mode='dyn' requires -cov/--coverage_pct")
        if not (0.0 < args.coverage_pct <= 100.0):
            raise ValueError(f"--coverage_pct must be in (0, 100], got {args.coverage_pct}")
        if args.coverage_min_k is not None and args.coverage_min_k <= 0:
            raise ValueError(f"--coverage_min_k must be > 0, got {args.coverage_min_k}")
        if args.coverage_max_k is not None and args.coverage_max_k <= 0:
            raise ValueError(f"--coverage_max_k must be > 0, got {args.coverage_max_k}")
        if (
            args.coverage_min_k is not None
            and args.coverage_max_k is not None
            and args.coverage_min_k > args.coverage_max_k
        ):
            raise ValueError("--coverage_min_k cannot exceed --coverage_max_k")

    if args.model_tag:
        model_tag = args.model_tag
    else:
        model_id = model_id_eff.split("/")[-1]
        model_tag = model_id.split("-")[0].lower() if "-" in model_id else model_id.lower()

    if args.mode == "hot":
        mode_tag = f"hotk{args.k}"
    elif args.mode == "dyn":
        mode_tag = f"cov{_coverage_tag(args.coverage_pct)}"
    elif args.mode == "random":
        mode_tag = f"randk{args.k}"
    else:
        mode_tag = "full_lora"
    run_name = f"{model_tag}_{args.task}_s{seed_eff}_{mode_tag}{args.run_name_suffix}"

    # Hotmap
    hotmap_path = None
    if args.mode in ("hot", "dyn"):
        if args.hotmap:
            hotmap_path = args.hotmap
        elif args.hotmap_template:
            hotmap_file = args.hotmap_template.format(
                task=args.task,
                mode=args.mode,
                k=args.k,
                coverage_pct=args.coverage_pct,
                seed=seed_eff,
                model=model_tag,
                run_name=run_name,
            )
            hotmap_path = os.path.join(args.hotmap_dir, hotmap_file) if args.hotmap_dir else hotmap_file
        elif args.telemetry:
            if args.mode == "dyn":
                hotmap_path = build_hotmap_by_coverage(
                    args.telemetry,
                    coverage_pct=args.coverage_pct,
                    mode=args.hotmap_mode,
                    min_k=args.coverage_min_k,
                    max_k=args.coverage_max_k,
                )
            else:
                hotmap_path = build_hotmap(args.telemetry, k=args.k, mode=args.hotmap_mode)
        else:
            raise ValueError(f"{args.mode} mode requires --hotmap, --hotmap_template, or --telemetry")

    # Hub repo resolution
    if args.push_to_hub:
        if args.hub_repo:
            repo_name = args.hub_repo
        else:
            repo_name = run_name

        try:
            info = HfApi().whoami()
            hub_user = info.get("name")
        except Exception:
            hub_user = None
        if not hub_user:
            raise ValueError("Could not detect HF username. Run `huggingface-cli login` or pass a valid token.")

        hub_repo = f"{hub_user}/{repo_name}"
    else:
        hub_repo = None

    # Launch
    print(f"Launching: {run_name}")
    run_training(
        model_id=model_id_eff,
        dataset_key=args.task,
        run_name=run_name,
        mode=args.mode,
        hotmap_json=hotmap_path,
        random_k=(args.k if args.mode == "random" else None),
        random_seed=args.random_seed,
        lr=args.lr,
        epochs=args.epochs,
        train_samples=args.train_samples,
        bs=args.bs,
        grad_acc=args.grad_acc,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        seed=args.seed,
        data_seed=args.data_seed,
        max_len=args.max_len,
        use_wandb=(not args.no_wandb),
        wandb_project=args.wandb_project,
        push_to_hub=args.push_to_hub,
        hub_repo=hub_repo,
        hub_private=args.hub_private,
        cleanup_after_push=args.cleanup_after_push,
    )

if __name__ == "__main__":
    main()
