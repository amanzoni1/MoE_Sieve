#!/usr/bin/env python3
import os
import json
import shutil
import time
import inspect
import torch
from typing import Optional, Dict, List, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi

try:
    import wandb
except ImportError:
    wandb = None

from .config import SYS_CFG, TRAIN_CFG
from .data_registry import DATASETS, load_and_format_dataset
from .utils_training import get_targets, infer_hot_k, build_random_hotmap, full_target_sanity


# Helpers
def save_run_artifacts(
    out_dir: str,
    run_cfg: Dict[str, Any],
    targets: List[str],
    hotmap_json: Optional[str],
    random_hotmap: Optional[Dict[str, List[int]]] = None,
):
    """Saves config and targets BEFORE training starts (Crash recovery)."""
    os.makedirs(out_dir, exist_ok=True)
    # Save Run Config
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)
    # Save Target Modules List
    with open(os.path.join(out_dir, "targets.json"), "w") as f:
        json.dump(targets, f, indent=2)
    # Backup the Hotmap used
    if hotmap_json and os.path.exists(hotmap_json):
        shutil.copy(hotmap_json, os.path.join(out_dir, "hotmap_used.json"))
    if random_hotmap is not None:
        with open(os.path.join(out_dir, "random_hotmap_used.json"), "w") as f:
            json.dump(random_hotmap, f, indent=2)


# MAIN TRAINING ROUTINE
def run_training(
    dataset_key: str,
    run_name: str,
    mode: str = "hot",
    hotmap_json: Optional[str] = None,
    random_k: Optional[int] = None,
    random_seed: Optional[int] = None,
    # Hyperparam Overrides
    lr: Optional[float] = None,
    epochs: Optional[int] = None,
    train_samples: Optional[int] = None,
    bs: Optional[int] = None,
    grad_acc: Optional[int] = None,
    r: Optional[int] = None,
    alpha: Optional[int] = None,
    dropout: Optional[float] = None,
    seed: Optional[int] = None,
    data_seed: Optional[int] = None,
    max_len: Optional[int] = None,
    # Misc
    use_wandb: Optional[bool] = None,
    wandb_project: Optional[str] = None,
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
    hub_private: bool = False,
    cleanup_after_push: bool = False,
):
    # Global Setup
    seed_eff = seed if seed is not None else TRAIN_CFG.seed
    set_seed(seed_eff)

    # TF32 (speed boost on Ada/Ampere)
    if TRAIN_CFG.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Paths & Configs
    out_dir = SYS_CFG.get_output_dir(f"runs/{dataset_key}/{run_name}")
    ds_cfg = DATASETS[dataset_key]

    epochs_eff = epochs if epochs is not None else ds_cfg.get("epochs", TRAIN_CFG.epochs)
    bs_eff = bs if bs is not None else TRAIN_CFG.per_device_bs
    grad_acc_eff = grad_acc if grad_acc is not None else TRAIN_CFG.grad_acc
    max_len_eff = max_len if max_len is not None else TRAIN_CFG.max_len
    lr_eff = lr if lr is not None else ds_cfg.get("lr", 4e-4)

    r_eff = r if r is not None else TRAIN_CFG.r
    alpha_eff = alpha if alpha is not None else TRAIN_CFG.alpha
    dropout_eff = dropout if dropout is not None else TRAIN_CFG.dropout

    random_seed_eff = random_seed if random_seed is not None else seed_eff
    data_seed_eff = data_seed if data_seed is not None else seed_eff

    if mode == "hot":
        hot_k = infer_hot_k(hotmap_json)
    elif mode == "random":
        hot_k = random_k
    else:
        hot_k = None

    use_wandb_eff = use_wandb if use_wandb is not None else TRAIN_CFG.use_wandb
    project_eff = wandb_project if wandb_project is not None else TRAIN_CFG.wandb_project

    if push_to_hub and not hub_repo:
        raise ValueError("push_to_hub=True requires hub_repo like 'username/repo_name'")

    print("\n" + "=" * 80)
    print("RUN CONFIG")
    print(f"  model_id:      {TRAIN_CFG.model_id}")
    print(f"  dataset:       {dataset_key}")
    print(f"  run_name:      {run_name}")
    print(f"  seed:          {seed_eff}")
    print(f"  data_seed:     {data_seed_eff}")
    print(f"  mode:          {mode}")
    print(f"  hotmap_json:   {hotmap_json}")
    print(f"  hot_k:         {hot_k}")
    if mode == "random":
        print(f"  random_seed:   {random_seed_eff}")
    print(f"  lr:            {lr_eff}")
    print(f"  epochs:        {epochs_eff}")
    print(f"  bs:            {bs_eff}")
    print(f"  grad_acc:      {grad_acc_eff}")
    print(f"  max_len:       {max_len_eff}")
    print(f"  train_samples: {train_samples}")
    print(f"  lora:          r={r_eff}, alpha={alpha_eff}, dropout={dropout_eff}")
    print(f"  wandb:         {use_wandb_eff} (project={project_eff})")
    print(f"  push_to_hub:   {push_to_hub} (repo={hub_repo}, private={hub_private})")
    print("=" * 80 + "\n")

    # W&B Init
    if use_wandb_eff and wandb is not None:
        wandb.init(
            project=project_eff,
            name=run_name,
            config={
                "model": TRAIN_CFG.model_id,
                "dataset": dataset_key,
                "run_name": run_name,
                "mode": mode,
                "seed": seed_eff,
                "data_seed": data_seed_eff,
                "max_len": max_len_eff,
                "bs_per_device": bs_eff,
                "grad_acc": grad_acc_eff,
                "epochs": epochs_eff,
                "learning_rate": lr_eff,
                "lora_r": r_eff,
                "lora_alpha": alpha_eff,
                "dropout": dropout_eff,
                "hotmap_path": hotmap_json,
                "hot_k": hot_k,
                "samples": train_samples or "full"
            }
        )

    # Model & Tokenizer
    print(f"Loading Model: {TRAIN_CFG.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_CFG.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TRAIN_CFG.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    # IMPORTANT: cache should be OFF when using gradient checkpointing
    if TRAIN_CFG.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    else:
        model.config.use_cache = TRAIN_CFG.use_cache

    if hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers
    num_layers = len(layers)
    num_experts = getattr(model.config, "num_experts", 64)

    # Target Selection
    targets = get_targets(
        model,
        mode,
        hotmap_json=hotmap_json,
        random_k=random_k,
        random_seed=random_seed_eff,
    )
    full_expected_trainable_params = None
    if mode == "full":
        sanity = full_target_sanity(
            model=model,
            suffixes=targets,
            num_layers=num_layers,
            num_experts=num_experts,
            lora_rank=r_eff,
        )
        print(f"   + Matched linear modules (FULL): {sanity['matched_count']}")
        print(f"   + Expected LoRA trainable params (FULL): {sanity['expected_trainable_params']:,}")
        full_expected_trainable_params = sanity["expected_trainable_params"]

    random_hotmap = None
    if mode == "random":
        if random_k is None:
            raise ValueError("mode='random' requires random_k")
        random_hotmap = build_random_hotmap(num_layers, num_experts, int(random_k), int(random_seed_eff))

    # PEFT Application
    peft_config = LoraConfig(
        r=r_eff,
        lora_alpha=alpha_eff,
        lora_dropout=dropout_eff,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets
    )
    model = get_peft_model(model, peft_config)

    # Log Trainable Params
    trainable, total = model.get_nb_trainable_parameters()
    if mode == "full" and full_expected_trainable_params is not None and trainable != full_expected_trainable_params:
        raise RuntimeError(
            f"Full-LoRA sanity check failed: trainable params {trainable:,} != expected {full_expected_trainable_params:,}. "
            "Target matching changed unexpectedly."
        )
    print(f"\n📊 [{dataset_key}/{run_name}] Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")
    if use_wandb_eff and wandb is not None:
        wandb.log({
            "trainable_params": trainable,
            "total_params": total,
            "trainable_frac": trainable/total
        })

    # Save Artifacts EARLY (Safety)
    save_run_artifacts(
        out_dir=out_dir,
        run_cfg={
            "dataset": dataset_key,
            "mode": mode,
            "model_id": TRAIN_CFG.model_id,
            "seed": seed_eff,
            "data_seed": data_seed_eff,
            "max_len": max_len_eff,
            "lr": lr_eff,
            "epochs": epochs_eff,
            "bs": bs_eff,
            "grad_acc": grad_acc_eff,
            "lora": {"r": r_eff, "alpha": alpha_eff, "dropout": dropout_eff},
            "hotmap_json": hotmap_json,
            "hot_k": hot_k,
            "random_seed": random_seed_eff if mode == "random" else None,
            "train_samples": train_samples,
        },
        targets=targets,
        hotmap_json=hotmap_json,
        random_hotmap=random_hotmap,
    )

    # Data Loading
    train_ds = load_and_format_dataset(
        dataset_key,
        tokenizer,
        max_len_eff,
        n_samples=train_samples,
        seed=seed_eff,
        data_seed=data_seed_eff,
    )

    # Training Arguments
    training_args_kwargs = dict(
        output_dir=out_dir,
        seed=seed_eff,
        per_device_train_batch_size=bs_eff,
        gradient_accumulation_steps=grad_acc_eff,
        learning_rate=lr_eff,
        num_train_epochs=epochs_eff,
        bf16=True,
        tf32=TRAIN_CFG.tf32,
        gradient_checkpointing=TRAIN_CFG.gradient_checkpointing,
        logging_steps=TRAIN_CFG.logging_steps,
        save_strategy="epoch",
        save_total_limit=TRAIN_CFG.save_total_limit,
        report_to="wandb" if use_wandb_eff else "none",
        remove_unused_columns=True,
    )
    # Keep HF Trainer/W&B config aligned with our effective dataset shuffle seed.
    if "data_seed" in inspect.signature(TrainingArguments).parameters:
        training_args_kwargs["data_seed"] = data_seed_eff

    args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Execute Training
    print("Training Started...")
    t0 = time.time()
    trainer.train()
    t1 = time.time()
    duration = round(t1 - t0, 1)
    print(f"Training Complete in {duration}s")
    if use_wandb_eff and wandb is not None:
        wandb.log({"train_seconds": duration})

    # Save Final Adapter
    final_path = os.path.join(out_dir, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved Final Adapter -> {final_path}")

    # Push to Hub
    if push_to_hub:
        api = HfApi()
        api.create_repo(repo_id=hub_repo, repo_type="model", exist_ok=True, private=hub_private)

        # Upload ONLY the final adapter folder content
        api.upload_folder(
            repo_id=hub_repo,
            repo_type="model",
            folder_path=final_path,
            path_in_repo=".",
        )
        print(f"[HF] Pushed final_adapter to: {hub_repo}")

        # Upload small run artifacts
        for fname in ("run_config.json", "targets.json", "hotmap_used.json"):
            local_path = os.path.join(out_dir, fname)
            if os.path.exists(local_path):
                api.upload_file(
                    repo_id=hub_repo,
                    repo_type="model",
                    path_or_fileobj=local_path,
                    path_in_repo=f"run_artifacts/{fname}",
                )
        print(f"[HF] Pushed run_artifacts to: {hub_repo}/run_artifacts/")

        if cleanup_after_push:
            # Remove the full run output directory to reclaim disk after a successful push
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir, ignore_errors=True)
                print(f"[Cleanup] Deleted local run dir: {out_dir}")

    if use_wandb_eff and wandb is not None:
        wandb.finish()

    # Cleanup
    del trainer, model
    torch.cuda.empty_cache()
