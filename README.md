# MoE-Sieve

Code, compact profiling and evaluation artifacts, a companion notebook, and the final paper PDF for [MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning](https://arxiv.org/abs/2603.24044).

MoE-Sieve profiles expert routing in Mixture-of-Experts language models and uses that signal to decide which experts receive LoRA adapters. Across OLMoE and Qwen on GSM8K, HellaSwag, and Spider, tuning only the top 25% most-routed experts per layer stays competitive with full LoRA while substantially reducing trainable parameters, checkpoint size, and training time.

## Paper

- arXiv: [2603.24044](https://arxiv.org/abs/2603.24044)
- PDF: `MoE_Sieve.pdf`
- Companion notebook: `moe_sieve_companion.ipynb`

## Repository Layout

- `src/`: core profiling, training, and evaluation logic
- `scripts/`: entry points for profiling, training, and evaluation
- `outputs/`: tracked reproducibility artifacts used by the paper
- `moe_sieve_companion.ipynb`: paper-facing notebook that loads the saved artifacts directly
- `MoE_Sieve.pdf`: final paper PDF

## Reproducibility Artifacts

The tracked `outputs/` tree keeps the compact files needed to inspect the paper results without re-running every experiment:

- `outputs/{model}/{task}/*_summary.json`: per-run metric summaries
- `outputs/telemetry/{model}/{dataset}/*.pt`: routing telemetry tensors
- `outputs/{model}/{task}/**/official_eval_output.txt`: canonical Spider Test Suite outputs

To rerun experiments, use `scripts/` as the CLI entry points and `src/` for the core implementation.

## Citation

```bibtex
@misc{manzoni2026moesieve,
  title={MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning},
  author={Andrea Manzoni},
  year={2026},
  eprint={2603.24044},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.24044}
}
```
