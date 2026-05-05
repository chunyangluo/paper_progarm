import os
import pandas as pd
import hashlib
import difflib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataFilter:
    """样本筛选模块，去除无效样本、重复样本，筛选高迷惑性边界样本"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self._create_directories()
    
    def _create_directories(self):
        """创建数据目录"""
        directories = [
            f"{self.data_dir}/processed/features",
            f"{self.data_dir}/processed/datasets"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_data(self, scenario, label):
        """加载数据"""
        # 加载主样本文件
        if label == "phishing":
            file_path = f"{self.data_dir}/raw/phishing/{scenario}/{scenario}_phishing.csv" if scenario != "link" else f"{self.data_dir}/raw/phishing/link/phishTank_links.csv"
        else:
            file_path = f"{self.data_dir}/raw/normal/{scenario}/{scenario}_normal.csv" if scenario != "link" else f"{self.data_dir}/raw/normal/link/alexa_links.csv"
        
        dfs = []
        if os.path.exists(file_path):
            dfs.append(pd.read_csv(file_path))
        
        # 尝试其他命名格式
        if label == "phishing":
            alt_path = f"{self.data_dir}/raw/phishing/{scenario}/phishing_{scenario}.csv"
        else:
            alt_path = f"{self.data_dir}/raw/normal/{scenario}/normal_{scenario}.csv"
        
        if os.path.exists(alt_path):
            dfs.append(pd.read_csv(alt_path))
        
        # 加载边界样本
        if label == "phishing":
            boundary_path = f"{self.data_dir}/raw/phishing/{scenario}/phishing_boundary.csv"
        else:
            boundary_path = f"{self.data_dir}/raw/normal/{scenario}/normal_boundary.csv"
        
        if os.path.exists(boundary_path):
            dfs.append(pd.read_csv(boundary_path))
        
        # 加载混淆样本
        if label == "phishing":
            confusion_path = f"{self.data_dir}/raw/phishing/{scenario}/phishing_confusion.csv"
        else:
            confusion_path = f"{self.data_dir}/raw/normal/{scenario}/normal_confusion.csv"
        
        if os.path.exists(confusion_path):
            dfs.append(pd.read_csv(confusion_path))
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def remove_invalid_samples(self, df):
        """去除无效样本"""
        print("正在去除无效样本...")
        
        # 去除空文本样本
        if 'text' in df.columns:
            df = df[df['text'].notna() & (df['text'].str.strip() != '')]
        
        # 去除空URL样本（对于链接场景）
        if 'url' in df.columns:
            df = df[df['url'].notna() & (df['url'].str.strip() != '')]
        
        # 去除格式错误的URL
        if 'url' in df.columns:
            url_pattern = r'^https?://[\w\-]+(\.[\w\-]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?$'
            df = df[df['url'].apply(lambda x: bool(re.match(url_pattern, str(x))))]
        
        print(f"去除无效样本后，剩余 {len(df)} 条样本")
        return df
    
    def remove_duplicates(self, df):
        """去除重复样本"""
        print("正在去除重复样本...")
        
        # 基于文本和URL生成唯一哈希值
        def generate_hash(row):
            text = str(row.get('text', ''))
            url = str(row.get('url', ''))
            combined = f"{text}_{url}"
            return hashlib.md5(combined.encode()).hexdigest()
        
        df['hash'] = df.apply(generate_hash, axis=1)
        df = df.drop_duplicates(subset=['hash'])
        df = df.drop('hash', axis=1)
        
        print(f"去除重复样本后，剩余 {len(df)} 条样本")
        return df
    
    def filter_boundary_samples(self, phishing_df, normal_df, threshold=0.7):
        """筛选高迷惑性边界样本"""
        print("正在筛选高迷惑性边界样本...")
        
        # 确保两数据集都有文本列
        if 'text' not in phishing_df.columns or 'text' not in normal_df.columns:
            print("文本列不存在，无法筛选边界样本")
            return phishing_df
        
        # 合并文本数据
        phishing_texts = phishing_df['text'].tolist()
        normal_texts = normal_df['text'].tolist()
        all_texts = phishing_texts + normal_texts
        
        # 使用TF-IDF向量化
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 计算钓鱼样本与正常样本的相似度
        phishing_vectors = tfidf_matrix[:len(phishing_texts)]
        normal_vectors = tfidf_matrix[len(phishing_texts):]
        
        # 计算每个钓鱼样本与最相似的正常样本的相似度
        boundary_samples = []
        for i, phishing_vector in enumerate(phishing_vectors):
            similarities = cosine_similarity(phishing_vector, normal_vectors)[0]
            max_similarity = max(similarities)
            
            if max_similarity >= threshold:
                boundary_samples.append(phishing_df.iloc[i])
        
        boundary_df = pd.DataFrame(boundary_samples)
        print(f"筛选出 {len(boundary_df)} 条高迷惑性边界样本")
        return boundary_df
    
    def supplement_high_frequency_scenarios(self, df, scenario):
        """补充高频场景样本"""
        print("正在补充高频场景样本...")
        
        # 高频场景关键词
        scenarios = {
            "政务": ["社保", "医保", "税务", "政府", "政务", "派出所", "公安局"],
            "快递": ["快递", "物流", "包裹", "顺丰", "圆通", "中通", "韵达", "申通", "EMS"],
            "金融": ["银行", "信用卡", "贷款", "理财", "保险", "投资", "股票", "基金"],
            "社交": ["微信", "QQ", "微博", "抖音", "快手", "知乎", "小红书", "B站"]
        }
        
        # 统计各场景样本数量
        scenario_counts = {}
        for scenario_name, keywords in scenarios.items():
            count = 0
            for keyword in keywords:
                if 'text' in df.columns:
                    count += df['text'].str.contains(keyword).sum()
            scenario_counts[scenario_name] = count
        
        print(f"当前各场景样本数量: {scenario_counts}")
        
        # 这里可以根据需要补充样本，目前只是统计
        # 实际应用中，可以根据场景分布情况，生成或采集更多样本
        
        return df
    
    def process_all(self):
        """处理所有样本"""
        print("开始处理所有样本...")
        
        scenarios = ["sms", "email", "link"]
        processed_data = {}
        
        for scenario in scenarios:
            print(f"\n处理 {scenario} 场景...")
            
            # 加载钓鱼样本
            phishing_df = self.load_data(scenario, "phishing")
            print(f"原始钓鱼样本数量: {len(phishing_df)}")
            
            # 加载正常样本
            normal_df = self.load_data(scenario, "normal")
            print(f"原始正常样本数量: {len(normal_df)}")
            
            # 去除无效样本
            phishing_df = self.remove_invalid_samples(phishing_df)
            normal_df = self.remove_invalid_samples(normal_df)
            
            # 去除重复样本
            phishing_df = self.remove_duplicates(phishing_df)
            normal_df = self.remove_duplicates(normal_df)
            
            # 筛选边界样本
            if scenario != "link":  # 链接场景没有文本，无法筛选边界样本
                boundary_df = self.filter_boundary_samples(phishing_df, normal_df)
                
                # 保存边界样本
                if len(boundary_df) > 0:
                    boundary_df.to_csv(f"{self.data_dir}/processed/boundary_{scenario}.csv", index=False, encoding="utf-8-sig")
            
            # 补充高频场景样本
            phishing_df = self.supplement_high_frequency_scenarios(phishing_df, scenario)
            normal_df = self.supplement_high_frequency_scenarios(normal_df, scenario)
            
            # 保存处理后的数据
            phishing_df.to_csv(f"{self.data_dir}/processed/phishing_{scenario}.csv", index=False, encoding="utf-8-sig")
            normal_df.to_csv(f"{self.data_dir}/processed/normal_{scenario}.csv", index=False, encoding="utf-8-sig")
            
            processed_data[scenario] = {
                "phishing": phishing_df,
                "normal": normal_df
            }
        
        print("\n样本处理完成！")
        return processed_data

if __name__ == "__main__":
    filter = DataFilter()
    filter.process_all()
