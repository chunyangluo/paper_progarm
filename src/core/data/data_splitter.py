from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def split_dataset(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    分割数据集为训练集、验证集和测试集
    
    参数:
        X: 特征数据
        y: 标签数据
        test_size: 测试集比例
        val_size: 验证集比例（相对于训练集）
        random_state: 随机种子
    
    返回:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    try:
        logger.info(f"分割数据集，测试集比例: {test_size}, 验证集比例: {val_size}")
        
        # 首先分割训练集和测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 然后从训练集中分割出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        
        logger.info(f"数据集分割完成: 训练集 {len(X_train)} 条, 验证集 {len(X_val)} 条, 测试集 {len(X_test)} 条")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logger.error(f"分割数据集失败: {e}")
        raise
