import pandas as pd
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

def augment_data(df, augment_factor=2, random_state=42):
    """
    数据增强
    
    参数:
        df: 输入DataFrame
        augment_factor: 增强因子
        random_state: 随机种子
    
    返回:
        DataFrame: 增强后的数据集
    """
    try:
        logger.info(f"开始数据增强，增强因子: {augment_factor}")
        
        # 设置随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 复制原始数据
        augmented_df = df.copy()
        
        # 生成增强数据
        augmented_rows = []
        for _, row in df.iterrows():
            for _ in range(augment_factor):
                new_row = row.copy()
                
                # 对文本进行简单的增强
                if 'text' in new_row:
                    text = new_row['text']
                    if text and isinstance(text, str):
                        # 随机插入空格
                        if len(text) > 5:
                            pos = random.randint(1, len(text)-1)
                            new_text = text[:pos] + ' ' + text[pos:]
                            new_row['text'] = new_text
                
                # 对URL进行简单的增强
                if 'url' in new_row:
                    url = new_row['url']
                    if url and isinstance(url, str):
                        # 生成URL变体
                        if '?' in url:
                            base, params = url.split('?', 1)
                            if '&' in params:
                                param_list = params.split('&')
                                if len(param_list) > 1:
                                    random.shuffle(param_list)
                                    new_params = '&'.join(param_list)
                                    new_row['url'] = f"{base}?{new_params}"
                
                augmented_rows.append(new_row)
        
        # 添加增强数据
        augmented_df = pd.concat([augmented_df, pd.DataFrame(augmented_rows)], ignore_index=True)
        
        # 打乱顺序
        augmented_df = augmented_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        logger.info(f"数据增强完成，原始数据 {len(df)} 条，增强后 {len(augmented_df)} 条")
        
        return augmented_df
    except Exception as e:
        logger.error(f"数据增强失败: {e}")
        raise
