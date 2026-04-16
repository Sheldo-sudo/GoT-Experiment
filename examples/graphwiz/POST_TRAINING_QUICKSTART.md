## GraphWiz Post-Training Quickstart

### 1) Run evaluation with trajectory export

```bash
python examples/graphwiz/graphwiz_eval_generic.py --source test --subset connectivity --lm_name qwen_local_4b --max_samples 20 --export_parquet 1
```

This creates:
- `sample_stats.csv`
- `trajectories.jsonl`
- optional `trajectories.parquet`

### 2) Build SFT / DPO / RL datasets

```bash
python examples/graphwiz/export_sft_data.py --input examples/graphwiz/results/<run_dir> --output examples/graphwiz/data/graphwiz_sft.jsonl --min_reward 0.4
python examples/graphwiz/export_pref_data.py --inputs examples/graphwiz/results/<run_dir1>,examples/graphwiz/results/<run_dir2> --output examples/graphwiz/data/graphwiz_pref.jsonl
python examples/graphwiz/export_rl_data.py --input examples/graphwiz/results/<run_dir> --output examples/graphwiz/data/graphwiz_rl.jsonl
```

### 3) Download local Qwen safetensors

```bash
python examples/graphwiz/download_qwen.py --repo_id Qwen/Qwen2.5-3B-Instruct --local_dir ./models/Qwen2.5-3B-Instruct
```

### 4) Minimal TRL SFT run (example)

Use `examples/graphwiz/train_configs/trl_sft_example.yaml` as a starter template.

### 5) LLaMA-Factory DPO dataset mapping

Use `examples/graphwiz/train_configs/llamafactory_dpo_dataset_info.json` and point it to the generated preference jsonl.
