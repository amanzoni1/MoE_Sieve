import argparse
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
    parser.add_argument("--hotmap_mode", type=str, default="counts", choices=["counts", "mass"])

    # --- Hyperparams (Default None = Use Registry/Config) ---
    parser.add_argument("--lr", type=float, default=None, help="Override Registry LR")
    parser.add_argument("--epochs", type=int, default=None, help="Override Registry Epochs")
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

    args = parser.parse_args()
    run_name = f"{args.task}_{args.mode}" + (f"_k{args.k}" if args.mode == "hot" else "_lora")

    # Hotmap
    hotmap_path = None
    if args.mode == "hot":
        if args.hotmap:
            hotmap_path = args.hotmap
        elif args.telemetry:
            hotmap_path = build_hotmap(args.telemetry, k=args.k, mode=args.hotmap_mode)
        else:
            raise ValueError("hot mode requires --hotmap or --telemetry")

    # Launch
    print(f"Launching: {run_name}")
    run_training(
        dataset_key=args.task,
        run_name=run_name,
        mode=args.mode,
        hotmap_json=hotmap_path,
        lr=args.lr,
        epochs=args.epochs,
        bs=args.bs,
        grad_acc=args.grad_acc,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        seed=args.seed,
        max_len=args.max_len,
        use_wandb=(not args.no_wandb),
        wandb_project=args.wandb_project
    )

if __name__ == "__main__":
    main()
