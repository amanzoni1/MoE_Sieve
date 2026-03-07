import argparse
import os
import torch
from huggingface_hub import HfApi
from src.utils_profiling import build_hotmap, build_hotmap_by_coverage, build_hotmap_by_budget
from src.trainer import run_training
from src.data_registry import DATASETS
from src.config import TRAIN_CFG


def _coverage_tag(coverage_pct: float) -> str:
    return f"{coverage_pct:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _budget_tag(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _telemetry_num_experts(pt_path: str) -> int:
    d = torch.load(pt_path, map_location="cpu")
    meta = d.get("meta", {}) if isinstance(d, dict) else {}
    n = meta.get("experts")
    if isinstance(n, (int, float)) and int(n) > 0:
        return int(n)
    layer_keys = sorted(k for k in d.keys() if isinstance(k, int)) if isinstance(d, dict) else []
    if not layer_keys:
        raise ValueError(f"Could not infer experts from telemetry: {pt_path}")
    first = d[layer_keys[0]]
    t = first.get("counts")
    if t is None:
        raise ValueError(f"Telemetry layer missing counts: {pt_path}")
    if t.dim() == 2:
        return int(t.shape[1])
    if t.dim() == 1:
        return int(t.shape[0])
    raise ValueError(f"Unsupported counts shape in telemetry: {tuple(t.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Experiment Launcher")

    # --- Essentials ---
    parser.add_argument("--task", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--mode", type=str, default="hot", choices=["hot", "dyn", "budget", "full", "random"])
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
    parser.add_argument(
        "--budget_per_layer",
        type=float,
        default=None,
        help="Target average routed experts per layer for mode=budget (e.g. 15).",
    )
    parser.add_argument(
        "--total_budget",
        type=int,
        default=None,
        help="Exact total routed expert slots for mode=budget (overrides --budget_per_layer).",
    )
    parser.add_argument(
        "--budget_frac",
        type=float,
        default=None,
        help="Fraction of routed experts per layer for mode=budget (e.g. 0.25 or 25).",
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
    if args.mode == "budget":
        has_prebuilt_hotmap = bool(args.hotmap or args.hotmap_template)
        if (
            args.total_budget is None
            and args.budget_per_layer is None
            and args.budget_frac is None
            and not has_prebuilt_hotmap
        ):
            raise ValueError(
                "mode='budget' requires one of --total_budget/--budget_per_layer/--budget_frac unless using --hotmap/--hotmap_template"
            )
        if args.budget_frac is not None and (args.total_budget is not None or args.budget_per_layer is not None):
            raise ValueError("--budget_frac cannot be combined with --total_budget or --budget_per_layer")
        if args.total_budget is not None and args.total_budget <= 0:
            raise ValueError(f"--total_budget must be > 0, got {args.total_budget}")
        if args.budget_per_layer is not None and args.budget_per_layer <= 0:
            raise ValueError(f"--budget_per_layer must be > 0, got {args.budget_per_layer}")
        if args.budget_frac is not None and args.budget_frac <= 0:
            raise ValueError(f"--budget_frac must be > 0, got {args.budget_frac}")
        if args.coverage_min_k is not None and args.coverage_min_k < 0:
            raise ValueError(f"--coverage_min_k must be >= 0 in budget mode, got {args.coverage_min_k}")
        if args.coverage_max_k is not None and args.coverage_max_k <= 0:
            raise ValueError(f"--coverage_max_k must be > 0, got {args.coverage_max_k}")
        if (
            args.coverage_min_k is not None
            and args.coverage_max_k is not None
            and args.coverage_min_k > args.coverage_max_k
        ):
            raise ValueError("--coverage_min_k cannot exceed --coverage_max_k")
        if args.coverage_min_k == 0:
            print(
                "⚠️  [Budget Mode] min_k=0 allows layers with zero routed expert adapters. "
                "This is valid, but can be surprising; attention/router/shared adapters are still trained."
            )

    budget_per_layer_eff = args.budget_per_layer
    total_budget_eff = args.total_budget
    budget_frac_eff = args.budget_frac
    if args.mode == "budget" and budget_frac_eff is not None:
        if not args.telemetry:
            raise ValueError("mode='budget' with --budget_frac requires --telemetry")
        frac = float(budget_frac_eff)
        if frac > 1.0:
            frac = frac / 100.0
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"--budget_frac must map to (0,1], got {budget_frac_eff}")
        n_experts = _telemetry_num_experts(args.telemetry)
        budget_per_layer_eff = frac * float(n_experts)
        budget_frac_eff = frac
        print(
            f"[Budget Mode] Auto budget from fraction: routed_experts={n_experts}, "
            f"budget_frac={frac:.4f} -> budget_per_layer={budget_per_layer_eff:.2f}"
        )

    if args.model_tag:
        model_tag = args.model_tag
    else:
        model_id = model_id_eff.split("/")[-1]
        model_tag = model_id.split("-")[0].lower() if "-" in model_id else model_id.lower()

    if args.mode == "hot":
        mode_tag = f"hotk{args.k}"
    elif args.mode == "dyn":
        mode_tag = f"cov{_coverage_tag(args.coverage_pct)}"
    elif args.mode == "budget":
        if total_budget_eff is not None:
            mode_tag = f"budget{total_budget_eff}"
        elif budget_frac_eff is not None:
            mode_tag = f"budgetp{_budget_tag(100.0 * budget_frac_eff)}"
        elif budget_per_layer_eff is not None:
            mode_tag = f"budgetk{_budget_tag(budget_per_layer_eff)}"
        else:
            mode_tag = "budget"
    elif args.mode == "random":
        mode_tag = f"randk{args.k}"
    else:
        mode_tag = "full_lora"
    run_name = f"{model_tag}_{args.task}_s{seed_eff}_{mode_tag}{args.run_name_suffix}"

    # Hotmap
    hotmap_path = None
    if args.mode in ("hot", "dyn", "budget"):
        if args.hotmap:
            hotmap_path = args.hotmap
        elif args.hotmap_template:
            hotmap_file = args.hotmap_template.format(
                task=args.task,
                mode=args.mode,
                k=args.k,
                coverage_pct=args.coverage_pct,
                budget_per_layer=budget_per_layer_eff,
                total_budget=total_budget_eff,
                budget_frac=budget_frac_eff,
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
            elif args.mode == "budget":
                hotmap_path = build_hotmap_by_budget(
                    args.telemetry,
                    budget_per_layer=budget_per_layer_eff,
                    total_budget=total_budget_eff,
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
    run_mode = "dyn" if args.mode == "budget" else args.mode
    run_training(
        model_id=model_id_eff,
        dataset_key=args.task,
        run_name=run_name,
        mode=run_mode,
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
