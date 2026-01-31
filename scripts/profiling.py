import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils_profiling import make_profile
from src.config import TRAIN_CFG
from src.data_registry import DATASETS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--n_samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--run_name", type=str, default="default", help="Tag for output file")
    args = parser.parse_args()

    # Load Model
    print(f"Loading model {TRAIN_CFG.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        TRAIN_CFG.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_CFG.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run Profile
    pt_file = make_profile(
        dataset_key=args.task,
        model=model,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        bs=args.bs,
        seed=args.seed,
        run_name=args.run_name
    )
    print(f"Telemetry at: {pt_file}")

if __name__ == "__main__":
    main()
