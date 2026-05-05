import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'versions')
SPLIT_DIR = os.path.join(PROJECT_ROOT, 'data', 'splits')
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, 'output', 'experiments')
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

SEED = 42
RATIO = (0.7, 0.2, 0.1)


def stratified_split(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=df['label'], random_state=seed
    )
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_ratio_adjusted, stratify=train_val['label'], random_state=seed
    )
    return train, val, test


def _resolve_dataset_path(dataset_name='dataset_20260421_100k.csv'):
    if os.path.isabs(dataset_name):
        return dataset_name
    if os.path.exists(dataset_name):
        return dataset_name
    return os.path.join(DATA_DIR, dataset_name)


def create_fixed_splits(dataset_name='dataset_20260421_100k.csv'):
    logger.info("=" * 60)
    logger.info("  Creating Fixed Data Splits (7:2:1)")
    logger.info("=" * 60)

    dataset_path = _resolve_dataset_path(dataset_name)
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return None

    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset: {len(df)} samples")

    # Robust preprocessing for split generation.
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str).str.strip()
    # Filter common invalid placeholders after astype(str).
    invalid_tokens = {"", "nan", "none", "null"}
    df = df[~df['text'].str.lower().isin(invalid_tokens)]
    df = df[df['text'].str.len() >= 2].reset_index(drop=True)

    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])].reset_index(drop=True)

    # Optional columns compatibility.
    if 'url' not in df.columns:
        df['url'] = ''
    else:
        df['url'] = df['url'].fillna('')

    if 'scenario' not in df.columns:
        df['scenario'] = 'general'
    else:
        df['scenario'] = df['scenario'].fillna('general')

    train, val, test = stratified_split(df, *RATIO, seed=SEED)

    logger.info(f"Train: {len(train)} (phishing={int((train['label']==1).sum())}, normal={int((train['label']==0).sum())})")
    logger.info(f"Val:   {len(val)} (phishing={int((val['label']==1).sum())}, normal={int((val['label']==0).sum())})")
    logger.info(f"Test:  {len(test)} (phishing={int((test['label']==1).sum())}, normal={int((test['label']==0).sum())})")

    train_path = os.path.join(SPLIT_DIR, 'train.csv')
    val_path = os.path.join(SPLIT_DIR, 'val.csv')
    test_path = os.path.join(SPLIT_DIR, 'test.csv')

    train.to_csv(train_path, index=False, encoding='utf-8')
    val.to_csv(val_path, index=False, encoding='utf-8')
    test.to_csv(test_path, index=False, encoding='utf-8')

    logger.info(f"Saved: {train_path}, {val_path}, {test_path}")

    split_info = {
        'dataset': dataset_name,
        'dataset_path': dataset_path,
        'total': len(df),
        'train': {'count': len(train), 'phishing': int((train['label']==1).sum()), 'normal': int((train['label']==0).sum())},
        'val': {'count': len(val), 'phishing': int((val['label']==1).sum()), 'normal': int((val['label']==0).sum())},
        'test': {'count': len(test), 'phishing': int((test['label']==1).sum()), 'normal': int((test['label']==0).sum())},
        'seed': SEED,
        'ratio': RATIO,
        'created_at': datetime.now().isoformat(),
    }
    info_path = os.path.join(SPLIT_DIR, 'split_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    return train, val, test


def load_splits(dataset_name='dataset_20260421_100k.csv'):
    train_path = os.path.join(SPLIT_DIR, 'train.csv')
    val_path = os.path.join(SPLIT_DIR, 'val.csv')
    test_path = os.path.join(SPLIT_DIR, 'test.csv')

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        logger.info("Fixed splits not found, creating...")
        return create_fixed_splits(dataset_name)

    # Validate split metadata matches requested dataset.
    info_path = os.path.join(SPLIT_DIR, 'split_info.json')
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                split_info = json.load(f)
            expected_dataset = split_info.get('dataset')
            if expected_dataset and expected_dataset != dataset_name:
                logger.warning(
                    f"Split dataset mismatch detected ({expected_dataset}). "
                    f"Rebuilding for {dataset_name}."
                )
                return create_fixed_splits(dataset_name)
        except Exception as e:
            logger.warning(f"Failed to read split_info.json, rebuilding splits: {e}")
            return create_fixed_splits(dataset_name)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    logger.info(f"Loaded fixed splits: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


if __name__ == '__main__':
    create_fixed_splits()
