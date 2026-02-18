import os
from dataclasses import dataclass

PROJECT_SLUG = "MoE_Sieve_Experiments"

@dataclass
class SystemConfig:
    @property
    def IS_COLAB(self) -> bool:
        return "COLAB_GPU" in os.environ

    # Paths
    @property
    def ROOT_DIR(self):
        env_root = os.getenv("MOE_SIEVE_ROOT_DIR")
        if env_root:
            return env_root

        if self.IS_COLAB:
            from google.colab import drive
            if not os.path.exists("/content/drive"):
                drive.mount("/content/drive")
            return f"/content/drive/MyDrive/{PROJECT_SLUG}"
        else:
            # RunPod / Local
            return f"/workspace/{PROJECT_SLUG}" if os.path.exists("/workspace") else f"./{PROJECT_SLUG}"

    def get_output_dir(self, subdir: str):
        path = os.path.join(self.ROOT_DIR, subdir)
        os.makedirs(path, exist_ok=True)
        return path

@dataclass
class TrainConfig:
    # --- Static Project Defaults ---
    model_id: str = "allenai/OLMoE-1B-7B-0924"
    seed: int = 42
    max_len: int = 2048

    # --- Hardware / Paper Training Defaults ---
    per_device_bs: int = 8
    grad_acc: int = 8
    gradient_checkpointing: bool = False
    use_cache: bool = False
    tf32: bool = True

    # --- Fallback Training Schedules ---
    epochs: int = 3
    logging_steps: int = 23
    save_total_limit: int = 2

    # --- LoRA Hyperparameters ---
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05

    # --- Tracking ---
    use_wandb: bool = True
    wandb_project: str = "moe-sieve"

SYS_CFG = SystemConfig()
TRAIN_CFG = TrainConfig()
