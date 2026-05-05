import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import json
import time
import random
import logging
import argparse
import subprocess
import numpy as np
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'experiment'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from data_split import load_splits, SPLIT_DIR, EXPERIMENT_DIR
from models import (TfidfLRModel, TextCNNExperiment, BERTExperiment,
                    LSTMExperiment, BERTTextCNNExperiment, MultimodalAblationExperiment)

NUM_REPEATS = 3
SEEDS = [42, 123, 456]
AVAILABLE_MODELS = [
    'TF-IDF+LR', 'TextCNN', 'BERT', 'LSTM', 'BERT-TextCNN',
    'Multimodal-text_only', 'Multimodal-text_url', 'Multimodal-text_network', 'Multimodal-full'
]
TRAINING_PROFILES = {
    # The profile changes model batch/epoch settings and the worker thread budget.
    # All profiles keep subprocess isolation enabled by default for crash containment.
    "safe": {
        "threads": 1,
        "bert_batch_size": 4,
        "bert_epochs": 15,
        "bert_patience": 5,
        "hybrid_batch_size": 8,
        "hybrid_grad_accum": 4,
        "hybrid_epochs": 15,
        "hybrid_patience": 5,
        "classic_batch_size": 64,
        "classic_epochs": 30,
        "classic_patience": 5,
    },
    "balanced": {
        "threads": 2,
        "bert_batch_size": 8,
        "bert_epochs": 12,
        "bert_patience": 4,
        "hybrid_batch_size": 12,
        "hybrid_grad_accum": 3,
        "hybrid_epochs": 12,
        "hybrid_patience": 4,
        "classic_batch_size": 128,
        "classic_epochs": 24,
        "classic_patience": 4,
    },
    "performance": {
        "threads": 4,
        "bert_batch_size": 16,
        "bert_epochs": 10,
        "bert_patience": 3,
        "hybrid_batch_size": 16,
        "hybrid_grad_accum": 2,
        "hybrid_epochs": 10,
        "hybrid_patience": 3,
        "classic_batch_size": 256,
        "classic_epochs": 18,
        "classic_patience": 3,
    },
}


def _model_configs():
    return [
        ('TF-IDF+LR', TfidfLRModel),
        ('TextCNN', TextCNNExperiment),
        ('BERT', BERTExperiment),
        ('LSTM', LSTMExperiment),
        ('BERT-TextCNN', BERTTextCNNExperiment),
        ('Multimodal-text_only', lambda: MultimodalAblationExperiment('text_only')),
        ('Multimodal-text_url', lambda: MultimodalAblationExperiment('text_url')),
        ('Multimodal-text_network', lambda: MultimodalAblationExperiment('text_network')),
        ('Multimodal-full', lambda: MultimodalAblationExperiment('full')),
    ]


def _get_model_class(model_name):
    for name, model_class in _model_configs():
        if name == model_name:
            return model_class
    raise ValueError(f"Invalid model name: {model_name}. Available: {AVAILABLE_MODELS}")


def _profile_settings(profile_name):
    if profile_name not in TRAINING_PROFILES:
        raise ValueError(f"Invalid training profile: {profile_name}. Available: {list(TRAINING_PROFILES)}")
    return TRAINING_PROFILES[profile_name]


def _apply_runtime_profile(profile_name):
    settings = _profile_settings(profile_name)
    threads = int(settings["threads"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 2)))
    except RuntimeError:
        pass


def _param_config_for_model(model_name, profile_name, benchmark_mode=False):
    settings = _profile_settings(profile_name)
    if benchmark_mode:
        # Short runs measure stability and throughput without spending hours tuning.
        bert_epochs = hybrid_epochs = classic_epochs = 2
        patience = 2
    else:
        bert_epochs = int(settings["bert_epochs"])
        hybrid_epochs = int(settings["hybrid_epochs"])
        classic_epochs = int(settings["classic_epochs"])
        patience = None

    if model_name == "BERT":
        return {
            "bert_lr": [2e-5],
            "cls_lr": [1e-4],
            "batch_size": [int(settings["bert_batch_size"])],
            "epochs": bert_epochs,
            "patience": patience or int(settings["bert_patience"]),
            "max_length": 128,
        }
    if model_name == "BERT-TextCNN":
        return {
            "batch_size": int(settings["hybrid_batch_size"]),
            "grad_accum_steps": int(settings["hybrid_grad_accum"]),
            "epochs": hybrid_epochs,
            "patience": patience or int(settings["hybrid_patience"]),
        }
    if model_name.startswith("Multimodal-"):
        return {
            "batch_size": int(settings["hybrid_batch_size"]),
            "grad_accum_steps": int(settings["hybrid_grad_accum"]),
            "epochs": hybrid_epochs,
            "patience": patience or int(settings["hybrid_patience"]),
        }
    if model_name == "TextCNN":
        return {
            "filter_sizes": [(3, 4, 5)],
            "num_filters": [128],
            "lr": [1e-4],
            "batch_size": int(settings["classic_batch_size"]),
            "epochs": classic_epochs,
            "patience": patience or int(settings["classic_patience"]),
        }
    if model_name == "LSTM":
        return {
            "hidden_dim": [128],
            "num_layers": [1],
            "lr": [5e-4],
            "batch_size": int(settings["classic_batch_size"]),
            "epochs": classic_epochs,
            "patience": patience or int(settings["classic_patience"]),
        }
    return None


def _build_split_payload(train_df, val_df, test_df):
    return {
        "train": {
            "texts": train_df['text'].tolist(),
            "labels": train_df['label'].tolist(),
            "urls": train_df['url'].fillna('').tolist() if 'url' in train_df.columns else None,
            "network_features": train_df['network_features'].tolist() if 'network_features' in train_df.columns else None,
        },
        "val": {
            "texts": val_df['text'].tolist(),
            "labels": val_df['label'].tolist(),
            "urls": val_df['url'].fillna('').tolist() if 'url' in val_df.columns else None,
            "network_features": val_df['network_features'].tolist() if 'network_features' in val_df.columns else None,
        },
        "test": {
            "texts": test_df['text'].tolist(),
            "labels": test_df['label'].tolist(),
            "urls": test_df['url'].fillna('').tolist() if 'url' in test_df.columns else None,
            "network_features": test_df['network_features'].tolist() if 'network_features' in test_df.columns else None,
        },
    }


def run_single_experiment(model_name, model_class, split_payload, seed, **kwargs):
    training_profile = kwargs.get("training_profile", "balanced")
    benchmark_mode = bool(kwargs.get("benchmark_mode", False))
    _apply_runtime_profile(training_profile)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_texts = split_payload["train"]["texts"]
    train_labels = split_payload["train"]["labels"]
    val_texts = split_payload["val"]["texts"]
    val_labels = split_payload["val"]["labels"]
    test_texts = split_payload["test"]["texts"]
    test_labels = split_payload["test"]["labels"]

    train_urls = split_payload["train"]["urls"]
    val_urls = split_payload["val"]["urls"]
    test_urls = split_payload["test"]["urls"]
    train_network_features = split_payload["train"]["network_features"]
    val_network_features = split_payload["val"]["network_features"]
    test_network_features = split_payload["test"]["network_features"]

    model = model_class()
    param_config = _param_config_for_model(model_name, training_profile, benchmark_mode=benchmark_mode)
    if isinstance(param_config, dict) and (
        model_name == "BERT-TextCNN" or model_name.startswith("Multimodal-")
    ):
        param_config = {**param_config, "seed": seed}
    start_time = time.time()

    if model_name == 'BERT-TextCNN':
        best_config, best_val_f1 = model.train(
            train_texts, train_labels, val_texts, val_labels,
            train_urls=train_urls, val_urls=val_urls,
            param_config=param_config,
        )
        test_metrics = model.evaluate(test_texts, test_labels, urls=test_urls)
    elif model_name.startswith('Multimodal-'):
        run_context = {
            "session_dir": kwargs.get("session_dir"),
            "model_name": model_name,
            "dataset_name": kwargs.get("dataset_name"),
            "seed": seed,
        }
        best_config, best_val_f1 = model.train(
            train_texts, train_labels, val_texts, val_labels,
            train_urls=train_urls, val_urls=val_urls,
            train_network_features=train_network_features,
            val_network_features=val_network_features,
            param_config=param_config,
            run_context=run_context,
        )
        test_metrics = model.evaluate(
            test_texts, test_labels, urls=test_urls, network_features=test_network_features
        )
    elif model_name in ['TF-IDF+LR']:
        best_config, best_val_f1 = model.train(train_texts, train_labels, val_texts, val_labels)
        test_metrics = model.evaluate(test_texts, test_labels)
    else:
        best_config, best_val_f1 = model.train(
            train_texts, train_labels, val_texts, val_labels, param_grid=param_config
        )
        test_metrics = model.evaluate(test_texts, test_labels)

    elapsed = time.time() - start_time
    test_metrics['training_time'] = elapsed
    test_metrics['best_config'] = best_config
    test_metrics['best_val_f1'] = best_val_f1
    test_metrics['seed'] = seed
    test_metrics['training_profile'] = training_profile
    test_metrics['benchmark_mode'] = benchmark_mode

    logger.info(f"  [{model_name}] Seed={seed} | Test F1={test_metrics['f1_score']:.4f} | "
                f"Acc={test_metrics['accuracy']:.4f} | P={test_metrics['precision']:.4f} | "
                f"R={test_metrics['recall']:.4f} | FPR={test_metrics['fpr']:.4f} | "
                f"FNR={test_metrics['fnr']:.4f} | Time={elapsed:.1f}s")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_metrics


def _init_or_load_session(session_dir: str):
    os.makedirs(session_dir, exist_ok=True)
    checkpoint_path = os.path.join(session_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(f"Resuming experiment from checkpoint: {checkpoint_path}")
    else:
        state = {
            "all_results": {},
            "completed_pairs": [],
            "failed_pairs": {},
            "meta": {
                "created_at": datetime.now().isoformat(),
                "seeds": SEEDS,
                "num_repeats": NUM_REPEATS,
            },
        }
    return state, checkpoint_path


def _save_checkpoint(state: dict, checkpoint_path: str):
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=str)


def _pair_key(model_name: str, seed: int) -> str:
    return f"{model_name}__seed_{seed}"


def _stratified_sample(df, n, seed):
    if n <= 0 or len(df) <= n or 'label' not in df.columns:
        return df
    _, sampled = train_test_split(df, test_size=n, random_state=seed, stratify=df['label'])
    return sampled.reset_index(drop=True)


def _load_split_payload(dataset_name, sample_size=0):
    train_df, val_df, test_df = load_splits(dataset_name=dataset_name)
    if sample_size and sample_size > 0:
        train_df = _stratified_sample(train_df, min(sample_size, len(train_df)), seed=42)
        val_df = _stratified_sample(val_df, min(max(64, sample_size // 4), len(val_df)), seed=42)
        test_df = _stratified_sample(test_df, min(max(64, sample_size // 4), len(test_df)), seed=42)
        logger.warning(
            f"Sample mode enabled: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

    logger.info(f"\nDataset: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    logger.info(f"Test set: phishing={int((test_df['label']==1).sum())}, normal={int((test_df['label']==0).sum())}")
    return _build_split_payload(train_df, val_df, test_df)


def _run_single_worker(model_name, seed, dataset_name, sample_size, session_dir, output_path,
                       training_profile="balanced", benchmark_mode=False):
    split_payload = _load_split_payload(dataset_name=dataset_name, sample_size=sample_size)
    metrics = run_single_experiment(
        model_name,
        _get_model_class(model_name),
        split_payload,
        seed,
        session_dir=session_dir,
        dataset_name=dataset_name,
        training_profile=training_profile,
        benchmark_mode=benchmark_mode,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)


def _run_pair_in_subprocess(model_name, seed, dataset_name, sample_size, session_dir,
                            training_profile="balanced", benchmark_mode=False):
    os.makedirs(session_dir, exist_ok=True)
    worker_dir = os.path.join(session_dir, "worker_results")
    os.makedirs(worker_dir, exist_ok=True)
    output_path = os.path.join(worker_dir, f"{model_name.replace('/', '_')}__seed_{seed}.json")
    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker-model", model_name,
        "--worker-seed", str(seed),
        "--worker-output", output_path,
        "--session-dir", session_dir,
        "--dataset-name", dataset_name,
        "--training-profile", training_profile,
    ]
    if sample_size and sample_size > 0:
        cmd.extend(["--sample-size", str(sample_size)])
    if benchmark_mode:
        cmd.append("--worker-benchmark-mode")

    env = os.environ.copy()
    threads = str(_profile_settings(training_profile)["threads"])
    env.update({
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PYTHONUNBUFFERED": "1",
    })

    logger.info(f"  [{model_name}] Running isolated worker process for seed={seed}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"worker process exited with code {result.returncode}")
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_all_experiments(session_dir: str = None, retry_failed: bool = False,
                        selected_models=None, sample_size: int = 0,
                        dataset_name: str = 'dataset_20260421_100k.csv',
                        use_worker_process: bool = True,
                        training_profile: str = "balanced",
                        selected_seeds=None):
    _apply_runtime_profile(training_profile)
    seeds = [int(s) for s in (selected_seeds or SEEDS)]
    logger.info("=" * 70)
    logger.info("  COMPARATIVE EXPERIMENT: Baselines + True Multimodal Ablation")
    logger.info("  Models: TF-IDF+LR, TextCNN, BERT, LSTM, BERT-TextCNN, Multimodal-*")
    logger.info(f"  Repeats: {len(seeds)}, Seeds: {seeds}")
    logger.info(f"  Training profile: {training_profile}")
    logger.info("=" * 70)

    split_payload = None if use_worker_process else _load_split_payload(
        dataset_name=dataset_name,
        sample_size=sample_size,
    )

    models_config = _model_configs()

    if session_dir is None:
        session_dir = os.path.join(EXPERIMENT_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
    state, checkpoint_path = _init_or_load_session(session_dir)
    state.setdefault("meta", {})
    state["meta"]["seeds"] = seeds
    state["meta"]["num_repeats"] = len(seeds)
    all_results = state.get("all_results", {})
    completed_pairs = set(state.get("completed_pairs", []))
    failed_pairs = state.get("failed_pairs", {})

    if selected_models:
        selected_models = set(selected_models)
        models_config = [it for it in models_config if it[0] in selected_models]
        logger.info(f"Filtered models: {[m for m, _ in models_config]}")

    try:
        for model_name, model_class in models_config:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Model: {model_name}")
            logger.info(f"{'='*60}")

            repeat_results = all_results.get(model_name, [])
            for repeat_idx, seed in enumerate(seeds):
                pair = _pair_key(model_name, seed)
                if pair in completed_pairs:
                    logger.info(f"  --- Repeat {repeat_idx+1}/{len(seeds)} (seed={seed}) skipped (already completed) ---")
                    continue
                if (not retry_failed) and pair in failed_pairs:
                    logger.info(f"  --- Repeat {repeat_idx+1}/{len(seeds)} (seed={seed}) skipped (previously failed) ---")
                    continue
                logger.info(f"\n  --- Repeat {repeat_idx+1}/{len(seeds)} (seed={seed}) ---")
                try:
                    if use_worker_process:
                        metrics = _run_pair_in_subprocess(
                            model_name, seed, dataset_name, sample_size, session_dir,
                            training_profile=training_profile,
                        )
                    else:
                        metrics = run_single_experiment(
                            model_name, model_class, split_payload, seed,
                            session_dir=session_dir, dataset_name=dataset_name,
                            training_profile=training_profile,
                        )
                    repeat_results.append(metrics)
                    all_results[model_name] = repeat_results
                    completed_pairs.add(pair)
                    state["all_results"] = all_results
                    state["completed_pairs"] = sorted(completed_pairs)
                    if "failed_pairs" in state and pair in state["failed_pairs"]:
                        state["failed_pairs"].pop(pair, None)
                    _save_checkpoint(state, checkpoint_path)
                    save_results(all_results, session_dir=session_dir)
                except Exception as e:
                    logger.error(f"  [{model_name}] Repeat {repeat_idx+1} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    state.setdefault("failed_pairs", {})[pair] = str(e)
                    _save_checkpoint(state, checkpoint_path)

            if repeat_results:
                all_results[model_name] = repeat_results
                summary = summarize_results(model_name, repeat_results)
                logger.info(f"\n  [{model_name}] Summary:")
                for k, v in summary.items():
                    if isinstance(v, float):
                        logger.info(f"    {k}: {v:.4f}")
                    else:
                        logger.info(f"    {k}: {v}")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Progress has been checkpointed.")
        save_results(all_results, session_dir=session_dir)
        raise

    save_results(all_results, session_dir=session_dir)
    return all_results


def summarize_results(model_name, results):
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'fpr', 'fnr']
    if results and 'auc' in results[0]:
        metric_keys.append('auc')

    summary = {'model': model_name, 'num_repeats': len(results)}
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))

    times = [r.get('training_time', 0) for r in results]
    summary['training_time_mean'] = float(np.mean(times))

    return summary


def save_results(all_results, session_dir=None):
    if session_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(EXPERIMENT_DIR, timestamp)
    else:
        results_dir = session_dir
    os.makedirs(results_dir, exist_ok=True)

    raw_path = os.path.join(results_dir, 'raw_results.json')
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    summaries = {}
    for model_name, results in all_results.items():
        summaries[model_name] = summarize_results(model_name, results)

    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_dir}")

    comparison_table = generate_comparison_table(summaries)
    table_path = os.path.join(results_dir, 'comparison_table.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(comparison_table)
    logger.info(f"\n{comparison_table}")

    return results_dir


def benchmark_training_profiles(selected_models, dataset_name, sample_size, session_dir=None):
    if session_dir is None:
        session_dir = os.path.join(EXPERIMENT_DIR, "profile_benchmark_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(session_dir, exist_ok=True)

    models = selected_models or ["BERT", "BERT-TextCNN", "Multimodal-full"]
    seed = SEEDS[0]
    results = []
    logger.info("=" * 70)
    logger.info("  PROFILE BENCHMARK: short isolated runs before formal training")
    logger.info(f"  Models: {models}")
    logger.info(f"  Sample size: {sample_size}")
    logger.info("=" * 70)

    for profile_name in TRAINING_PROFILES:
        for model_name in models:
            if model_name not in AVAILABLE_MODELS:
                raise ValueError(f"Invalid model name: {model_name}. Available: {AVAILABLE_MODELS}")
            start = time.time()
            record = {
                "profile": profile_name,
                "model": model_name,
                "seed": seed,
                "sample_size": sample_size,
                "status": "running",
            }
            try:
                metrics = _run_pair_in_subprocess(
                    model_name,
                    seed,
                    dataset_name,
                    sample_size,
                    session_dir,
                    training_profile=profile_name,
                    benchmark_mode=True,
                )
                record.update({
                    "status": "ok",
                    "elapsed_seconds": time.time() - start,
                    "f1_score": metrics.get("f1_score"),
                    "accuracy": metrics.get("accuracy"),
                    "best_val_f1": metrics.get("best_val_f1"),
                    "best_config": metrics.get("best_config"),
                })
            except Exception as exc:
                record.update({
                    "status": "failed",
                    "elapsed_seconds": time.time() - start,
                    "error": str(exc),
                })
            results.append(record)
            benchmark_path = os.path.join(session_dir, "profile_benchmark.json")
            with open(benchmark_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Profile benchmark saved to: {os.path.join(session_dir, 'profile_benchmark.json')}")
    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        by_profile = {}
        for profile_name in TRAINING_PROFILES:
            rows = [r for r in ok_results if r["profile"] == profile_name]
            if rows:
                by_profile[profile_name] = {
                    "successes": len(rows),
                    "total_seconds": sum(r["elapsed_seconds"] for r in rows),
                    "mean_f1": float(np.mean([r.get("f1_score", 0.0) for r in rows])),
                }
        recommended = sorted(
            by_profile.items(),
            key=lambda it: (-it[1]["successes"], it[1]["total_seconds"], -it[1]["mean_f1"]),
        )[0][0]
        logger.info(f"Recommended profile: {recommended}")
    return results


def generate_comparison_table(summaries):
    models_order = [
        'TF-IDF+LR', 'TextCNN', 'BERT', 'LSTM', 'BERT-TextCNN',
        'Multimodal-text_only', 'Multimodal-text_url', 'Multimodal-text_network', 'Multimodal-full'
    ]
    metrics_order = ['accuracy', 'precision', 'recall', 'f1_score', 'fpr', 'fnr']

    header = f"{'Model':<20}"
    for m in metrics_order:
        header += f"  {m:<16}"
    header += f"  {'Time(s)':<10}"
    header += "\n" + "-" * (20 + 18 * len(metrics_order) + 12)

    rows = []
    for model_name in models_order:
        if model_name not in summaries:
            continue
        s = summaries[model_name]
        row = f"{model_name:<20}"
        for m in metrics_order:
            mean_key = f'{m}_mean'
            std_key = f'{m}_std'
            if mean_key in s:
                row += f"  {s[mean_key]:.4f}±{s[std_key]:.4f}  "
            else:
                row += f"  {'N/A':<16}"
        row += f"  {s.get('training_time_mean', 0):.1f}"
        rows.append(row)

    return header + "\n" + "\n".join(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run comparative experiments with resume support")
    parser.add_argument(
        "--session-dir",
        type=str,
        default="",
        help="Existing/new experiment session directory for checkpoint resume",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry pairs that failed in previous runs",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names to run (subset)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Debug mode: sample limited train rows and proportional val/test rows",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="dataset_20260421_100k.csv",
        help="Dataset filename under data/versions used for split consistency check",
    )
    parser.add_argument(
        "--no-worker-process",
        action="store_true",
        help="Run model/seed pairs in the main process instead of isolated worker processes",
    )
    parser.add_argument(
        "--training-profile",
        type=str,
        choices=list(TRAINING_PROFILES),
        default="balanced",
        help="Speed/stability profile for neural training",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds to run (defaults to 42,123,456)",
    )
    parser.add_argument(
        "--benchmark-profiles",
        action="store_true",
        help="Run short isolated profile benchmarks before formal training",
    )
    parser.add_argument("--worker-model", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-benchmark-mode", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_model:
        _run_single_worker(
            model_name=args.worker_model,
            seed=args.worker_seed,
            dataset_name=args.dataset_name,
            sample_size=args.sample_size,
            session_dir=args.session_dir or EXPERIMENT_DIR,
            output_path=args.worker_output,
            training_profile=args.training_profile,
            benchmark_mode=args.worker_benchmark_mode,
        )
        sys.exit(0)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else None
    if model_list:
        invalid = [m for m in model_list if m not in AVAILABLE_MODELS]
        if invalid:
            raise ValueError(f"Invalid model names: {invalid}. Available: {AVAILABLE_MODELS}")
    if args.benchmark_profiles:
        benchmark_training_profiles(
            selected_models=model_list,
            dataset_name=args.dataset_name,
            sample_size=args.sample_size or 512,
            session_dir=args.session_dir or None,
        )
        sys.exit(0)
    run_all_experiments(
        session_dir=args.session_dir or None,
        retry_failed=args.retry_failed,
        selected_models=model_list,
        sample_size=args.sample_size,
        dataset_name=args.dataset_name,
        use_worker_process=not args.no_worker_process,
        training_profile=args.training_profile,
        selected_seeds=seed_list,
    )
