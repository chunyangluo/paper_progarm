#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定期更新多模态数据集脚本
实现首次全量采集，后续增量更新
"""
import os
import sys
import time
import schedule
import logging
import argparse
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_full_collection():
    """首次全量采集"""
    logger.info("开始首次全量采集...")
    try:
        # 运行全量采集
        import build_real_multimodal_dataset
        df = build_real_multimodal_dataset.build_real_multimodal_dataset(rounds=3, cumulative=False)
        logger.info(f"全量采集完成，共 {len(df)} 条样本")
        
        # 采集网络特征
        success = build_real_multimodal_dataset.collect_network_features("../data/real_multimodal_dataset.csv")
        if success:
            logger.info("网络特征采集完成")
        else:
            logger.warning("网络特征采集失败，但数据集已更新")
            
    except Exception as e:
        logger.error(f"全量采集失败: {e}")

def run_incremental_update():
    """后续增量更新"""
    logger.info("开始增量更新...")
    try:
        # 运行增量采集（使用累积模式）
        import build_real_multimodal_dataset
        df = build_real_multimodal_dataset.build_real_multimodal_dataset(rounds=1, cumulative=True)
        logger.info(f"增量更新完成，共 {len(df)} 条样本")
        
        # 采集网络特征（只采集新增数据）
        # 这里可以实现只对新增URL采集网络特征的逻辑
        success = build_real_multimodal_dataset.collect_network_features("../data/real_multimodal_dataset.csv")
        if success:
            logger.info("网络特征采集完成")
        else:
            logger.warning("网络特征采集失败，但数据集已更新")
            
    except Exception as e:
        logger.error(f"增量更新失败: {e}")

def check_if_first_run():
    """检查是否是首次运行"""
    dataset_path = "../data/real_multimodal_dataset.csv"
    return not os.path.exists(dataset_path)

def job():
    """定时任务"""
    logger.info("执行定时更新任务...")
    if check_if_first_run():
        run_full_collection()
    else:
        run_incremental_update()
    logger.info("定时更新任务完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="定期更新多模态数据集")
    parser.add_argument("--run-now", action="store_true", help="立即运行一次采集")
    parser.add_argument("--schedule", type=str, default="daily", choices=["daily", "weekly", "hourly"], help="定时任务频率")
    args = parser.parse_args()
    
    if args.run_now:
        # 立即运行一次
        logger.info("立即运行采集任务...")
        if check_if_first_run():
            run_full_collection()
        else:
            run_incremental_update()
        logger.info("采集任务完成")
        return
    
    # 设置定时任务
    logger.info(f"设置{args.schedule}定时更新任务...")
    
    if args.schedule == "daily":
        schedule.every().day.at("00:00").do(job)
    elif args.schedule == "weekly":
        schedule.every().monday.at("00:00").do(job)
    elif args.schedule == "hourly":
        schedule.every().hour.do(job)
    
    logger.info("定时任务已设置，开始运行...")
    logger.info("按Ctrl+C退出")
    
    # 运行首次采集
    if check_if_first_run():
        run_full_collection()
    
    # 运行调度器
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
