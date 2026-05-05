#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集收集验证报告生成脚本
"""
import pandas as pd
import os
from datetime import datetime

def generate_dataset_report():
    print('='*80)
    print('Dataset Collection Status Verification Report')
    print('='*80)

    versions_dir = 'c:/Users/chuny/Desktop/paper_progarm/data/versions'
    datasets = {}
    for f in os.listdir(versions_dir):
        if f.endswith('.csv'):
            path = os.path.join(versions_dir, f)
            df = pd.read_csv(path)
            datasets[f] = {
                'total': len(df),
                'phishing': len(df[df.label==1]) if 'label' in df.columns else 0,
                'normal': len(df[df.label==0]) if 'label' in df.columns else 0,
                'columns': len(df.columns),
                'size_kb': os.path.getsize(path)/1024
            }

    print(f'\nDetected {len(datasets)} version datasets:\n')
    for name, stats in sorted(datasets.items()):
        print(f'  {name}')
        print(f'    Total: {stats["total"]}, Phishing: {stats["phishing"]}, Normal: {stats["normal"]}')
        print(f'    Columns: {stats["columns"]}, Size: {stats["size_kb"]:.1f} KB')

    latest = 'dataset_20260411_chifraud_with_network_features_async.csv'
    if latest in datasets:
        df = pd.read_csv(os.path.join(versions_dir, latest))
        print(f'\nLatest dataset [{latest}] details:')
        print(f'  Fields: {df.columns.tolist()}')

        if 'source' in df.columns:
            print(f'\n  Source distribution:')
            for src, cnt in df['source'].value_counts().items():
                print(f'    {src}: {cnt}')

        if 'scenario' in df.columns:
            print(f'\n  Scenario distribution:')
            for sc, cnt in df['scenario'].value_counts().items():
                print(f'    {sc}: {cnt}')

    print('\n' + '='*80)
    return datasets

if __name__ == '__main__':
    generate_dataset_report()
