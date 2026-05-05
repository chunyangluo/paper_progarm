import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataSplitter:
    """数据集划分与标注模块"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self._create_directories()
    
    def _create_directories(self):
        """创建数据目录"""
        directories = [
            f"{self.data_dir}/processed/datasets"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_processed_data(self):
        """加载处理后的数据"""
        scenarios = ["sms", "email", "link"]
        data = {}
        
        for scenario in scenarios:
            # 加载处理后的数据
            phishing_df = pd.read_csv(f"{self.data_dir}/processed/phishing_{scenario}.csv")
            normal_df = pd.read_csv(f"{self.data_dir}/processed/normal_{scenario}.csv")
            
            # 添加标签
            phishing_df['label'] = 1
            normal_df['label'] = 0
            
            # 添加场景信息
            phishing_df['scenario'] = scenario
            normal_df['scenario'] = scenario
            
            # 合并数据
            combined_df = pd.concat([phishing_df, normal_df], ignore_index=True)
            data[scenario] = combined_df
        
        # 合并所有场景的数据
        all_data = pd.concat(data.values(), ignore_index=True)
        return all_data, data
    
    def split_data(self, df, test_size=0.1, val_size=0.2, random_state=42):
        """划分数据集"""
        print("正在划分数据集...")
        
        # 首先划分训练集和测试集
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df['label']
        )
        
        # 然后从训练集中划分验证集
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_ratio, 
            random_state=random_state, 
            stratify=train_val_df['label']
        )
        
        print(f"数据集划分完成：")
        print(f"训练集: {len(train_df)} 条样本")
        print(f"验证集: {len(val_df)} 条样本")
        print(f"测试集: {len(test_df)} 条样本")
        
        # 检查样本分布
        print("\n样本分布:")
        print("训练集:")
        print(train_df['label'].value_counts())
        print("验证集:")
        print(val_df['label'].value_counts())
        print("测试集:")
        print(test_df['label'].value_counts())
        
        return train_df, val_df, test_df
    
    def process_all(self):
        """处理所有数据"""
        print("开始处理数据集划分...")
        
        # 加载处理后的数据
        all_data, scenario_data = self.load_processed_data()
        
        # 划分所有数据
        train_df, val_df, test_df = self.split_data(all_data)
        
        # 保存划分后的数据集
        train_df.to_csv(f"{self.data_dir}/processed/datasets/train.csv", index=False, encoding="utf-8-sig")
        val_df.to_csv(f"{self.data_dir}/processed/datasets/val.csv", index=False, encoding="utf-8-sig")
        test_df.to_csv(f"{self.data_dir}/processed/datasets/test.csv", index=False, encoding="utf-8-sig")
        
        # 按场景划分数据
        for scenario, df in scenario_data.items():
            train_df, val_df, test_df = self.split_data(df)
            train_df.to_csv(f"{self.data_dir}/processed/datasets/{scenario}_train.csv", index=False, encoding="utf-8-sig")
            val_df.to_csv(f"{self.data_dir}/processed/datasets/{scenario}_val.csv", index=False, encoding="utf-8-sig")
            test_df.to_csv(f"{self.data_dir}/processed/datasets/{scenario}_test.csv", index=False, encoding="utf-8-sig")
        
        print("\n数据集划分完成！")
        return train_df, val_df, test_df

if __name__ == "__main__":
    splitter = DataSplitter()
    splitter.process_all()
