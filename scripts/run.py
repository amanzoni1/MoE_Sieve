import argparse
import os
from src.utils_profiling import build_hotmap
from src.trainer import run_training
from src.data_registry import DATASETS
from src.config import TRAIN_CFG

def main():
    parser = argparse.ArgumentParser(description="HELLoRA Experiment Launcher")

    # --- Essentials ---
    parser.add_argument("--task", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--mode", type=str, default="hot", choices=["hot", "full"])

    # --- Hot Mode ---
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--telemetry", type=str, help="Path to telemetry .pt")
    parser.add_argument("--hotmap", type=str, help="Path to hotmap .json")
    parser.add_argument("--hotmap_dir", type=str, default=None, help="Optional dir for hotmap_template")
    parser.add_argument(
        "--hotmap_template",
        type=str,
        default=None,
        help="Filename template with {k}, {task}, {seed}, {mode}. Example: telemetry_{task}_train_n7473_seed123_global_hotmap_counts_k{k}.json",
    )
    parser.add_argument("--hotmap_mode", type=str, default="counts", choices=["counts", "mass"])

    # --- Hyperparams (Default None = Use Registry/Config) ---
    parser.add_argument("--lr", type=float, default=None, help="Override Registry LR")
    parser.add_argument("--epochs", type=int, default=None, help="Override Registry Epochs")
    parser.add_argument("--train_samples", type=int, default=None, help="Debug: limit train samples")
    parser.add_argument("--bs", type=int, default=None, help=f"Override Config BS ({TRAIN_CFG.per_device_bs})")
    parser.add_argument("--grad_acc", type=int, default=None, help=f"Override Config GradAcc ({TRAIN_CFG.grad_acc})")
    parser.add_argument("--seed", type=int, default=None, help=f"Override ({TRAIN_CFG.seed})")
    parser.add_argument("--max_len", type=int, default=None, help=f"Override ({TRAIN_CFG.max_len})")
    parser.add_argument("--r", type=int, default=None, help=f"Override ({TRAIN_CFG.r})")
    parser.add_argument("--alpha", type=int, default=None, help=f"Override ({TRAIN_CFG.alpha})")
    parser.add_argument("--dropout", type=float, default=None, help=f"Override ({TRAIN_CFG.dropout})")

    # --- Misc ---
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo", type=str, default=None, help="e.g. username/repo_name")
    parser.add_argument(
        "--hub_repo_template",
        type=str,
        default=None,
        help="Template with {k}, {task}, {seed}, {mode}. Example: AManzoni/hellora-olmoe-{task}_hot_k{k}_s{seed}",
    )
    parser.add_argument("--hub_private", action="store_true")
    parser.add_argument(
        "--cleanup_after_push",
        action="store_true",
        help="Delete local run outputs after successful Hub push",
    )

    args = parser.parse_args()
    run_name = f"{args.task}_{args.mode}" + (f"_k{args.k}" if args.mode == "hot" else "_lora")
    if args.seed is not None:
        run_name += f"_s{args.seed}"

    seed_eff = args.seed if args.seed is not None else TRAIN_CFG.seed

    def render_template(tpl: str) -> str:
        return tpl.format(task=args.task, mode=args.mode, k=args.k, seed=seed_eff)

    # Hotmap
    hotmap_path = None
    if args.mode == "hot":
        if args.hotmap:
            hotmap_path = args.hotmap
        elif args.hotmap_template:
            hotmap_file = render_template(args.hotmap_template)
            hotmap_path = os.path.join(args.hotmap_dir, hotmap_file) if args.hotmap_dir else hotmap_file
        elif args.telemetry:
            hotmap_path = build_hotmap(args.telemetry, k=args.k, mode=args.hotmap_mode)
        else:
            raise ValueError("hot mode requires --hotmap or --telemetry")

    # Hub repo templating
    hub_repo = args.hub_repo
    if args.push_to_hub and not hub_repo and args.hub_repo_template:
        hub_repo = render_template(args.hub_repo_template)

    # Launch
    print(f"Launching: {run_name}")
    run_training(
        dataset_key=args.task,
        run_name=run_name,
        mode=args.mode,
        hotmap_json=hotmap_path,
        lr=args.lr,
        epochs=args.epochs,
        train_samples=args.train_samples,
        bs=args.bs,
        grad_acc=args.grad_acc,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        seed=args.seed,
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
