import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def load_dataset(dataset_path):
    """
    加载数据集
    
    参数:
        dataset_path: 数据集路径
    
    返回:
        DataFrame: 加载的数据集
    """
    try:
        logger.info(f"加载数据集: {dataset_path}")
        df = pd.read_csv(dataset_path)
        logger.info(f"数据集加载成功，包含 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise

def preprocess_text(text):
    """
    预处理文本
    
    参数:
        text: 输入文本
    
    返回:
        str: 预处理后的文本
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text)
    # 去除多余的空白字符
    text = ' '.join(text.split())
    return text
