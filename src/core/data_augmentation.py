import os
import pandas as pd
import numpy as np
import random
import re
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

class DataAugmenter:
    """数据集增强与优化模块"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self._create_directories()
        self.synonyms = self._load_synonyms()
    
    def _create_directories(self):
        """创建数据目录"""
        directories = [
            f"{self.data_dir}/augmented"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_synonyms(self):
        """加载同义词词典"""
        # 简单的同义词词典
        synonyms = {
            "账户": ["账号", "账户信息", "用户账户"],
            "密码": ["口令", "密码信息", "登录密码"],
            "验证": ["核实", "确认", "验证信息"],
            "安全": ["安全防护", "安全保障", "安全措施"],
            "风险": ["危险", "风险提示", "风险警告"],
            "冻结": ["锁定", "停用", "禁用"],
            "支付": ["付款", "支付款项", "完成支付"],
            "订单": ["定单", "订单信息", "购物订单"],
            "快递": ["物流", "快件", "包裹"],
            "领取": ["接收", "收取", "获取"],
            "验证码": ["校验码", "验证代码", "安全码"],
            "充值": ["充值金额", "充值服务", "充值话费"],
            "积分": ["积分奖励", "积分兑换", "积分累计"],
            "会员": ["会员资格", "会员等级", "会员服务"],
            "中奖": ["获奖", "得奖", "中奖信息"],
            "社保": ["社会保险", "社保信息", "社保账户"],
            "医保": ["医疗保险", "医保信息", "医保账户"],
            "银行": ["银行账户", "银行业务", "银行服务"]
        }
        return synonyms
    
    def augment_text(self, text, num_aug=3):
        """文本增强"""
        augmented_texts = []
        
        if not text or pd.isna(text):
            return augmented_texts
        
        for i in range(num_aug):
            # 同义词替换
            augmented = self._synonym_replacement(text)
            # 语序调整
            augmented = self._shuffle_sentences(augmented)
            # 随机插入
            augmented = self._random_insertion(augmented)
            # 随机删除
            augmented = self._random_deletion(augmented)
            
            if augmented != text and augmented:
                augmented_texts.append(augmented)
        
        return augmented_texts
    
    def _synonym_replacement(self, text):
        """同义词替换"""
        words = list(text)
        new_words = words.copy()
        
        for i, word in enumerate(words):
            if word in self.synonyms:
                synonym = random.choice(self.synonyms[word])
                new_words[i] = synonym
        
        return ''.join(new_words)
    
    def _shuffle_sentences(self, text):
        """语序调整"""
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            random.shuffle(sentences)
            return '。'.join(sentences) + '。'
        else:
            return text
    
    def _random_insertion(self, text):
        """随机插入"""
        words = list(text)
        if len(words) < 5:
            return text
        
        insert_pos = random.randint(1, len(words) - 1)
        insert_word = random.choice(list(self.synonyms.keys()))
        words.insert(insert_pos, insert_word)
        
        return ''.join(words)
    
    def _random_deletion(self, text):
        """随机删除"""
        words = list(text)
        if len(words) < 3:
            return text
        
        delete_pos = random.randint(0, len(words) - 1)
        del words[delete_pos]
        
        return ''.join(words)
    
    def augment_url(self, url, num_aug=3):
        """URL增强"""
        augmented_urls = []
        
        if not url or pd.isna(url):
            return augmented_urls
        
        for i in range(num_aug):
            # 相似域名生成
            augmented = self._generate_similar_domain(url)
            # 参数扰动
            augmented = self._perturb_parameters(augmented)
            # 路径变形
            augmented = self._modify_path(augmented)
            
            if augmented != url and augmented:
                augmented_urls.append(augmented)
        
        return augmented_urls
    
    def _generate_similar_domain(self, url):
        """生成相似域名"""
        # 简单的相似域名生成
        domain_pattern = r'https?://([^/]+)/'
        match = re.search(domain_pattern, url)
        if match:
            domain = match.group(1)
            # 随机替换一个字符
            if len(domain) > 3:
                pos = random.randint(1, len(domain) - 2)
                new_char = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
                new_domain = domain[:pos] + new_char + domain[pos+1:]
                return url.replace(domain, new_domain)
        return url
    
    def _perturb_parameters(self, url):
        """参数扰动"""
        if '?' in url:
            base_url, params = url.split('?', 1)
            param_list = params.split('&')
            # 过滤出有效的参数（包含=的参数）
            valid_params = [p for p in param_list if '=' in p]
            # 随机修改一个参数值
            if valid_params:
                idx = random.randint(0, len(valid_params) - 1)
                key, value = valid_params[idx].split('=', 1)
                new_value = str(random.randint(100000, 999999))
                valid_params[idx] = f"{key}={new_value}"
                new_params = '&'.join(valid_params)
                return f"{base_url}?{new_params}"
        return url
    
    def _modify_path(self, url):
        """路径变形"""
        if '/' in url:
            parts = url.split('/')
            # 随机添加一个路径段
            if len(parts) > 3:
                pos = random.randint(3, len(parts) - 1)
                new_path = str(random.randint(1000, 9999))
                parts.insert(pos, new_path)
                return '/'.join(parts)
        return url
    
    def augment_network_features(self, features):
        """网络行为特征增强"""
        # 特征扰动
        augmented = features.copy()
        for i in range(len(augmented)):
            if random.random() > 0.7:
                # 对数值特征进行微小扰动
                augmented[i] *= (1 + random.uniform(-0.1, 0.1))
        return augmented
    
    def handle_class_imbalance(self, df):
        """处理样本不平衡问题"""
        print("正在处理样本不平衡问题...")
        
        # 分离类别
        phishing_df = df[df['label'] == 1]
        normal_df = df[df['label'] == 0]
        
        # 确定样本数量
        min_count = min(len(phishing_df), len(normal_df))
        
        # 过采样少数类
        if len(phishing_df) < len(normal_df):
            phishing_df = resample(phishing_df, replace=True, n_samples=len(normal_df), random_state=42)
        else:
            normal_df = resample(normal_df, replace=True, n_samples=len(phishing_df), random_state=42)
        
        # 合并数据
        balanced_df = pd.concat([phishing_df, normal_df], ignore_index=True)
        
        print(f"处理后样本分布:")
        print(balanced_df['label'].value_counts())
        
        return balanced_df
    
    def augment_dataset(self, df, num_aug=3):
        """增强数据集"""
        print("正在增强数据集...")
        
        augmented_samples = []
        
        try:
            for _, row in df.iterrows():
                try:
                    # 添加原始样本
                    augmented_samples.append(row.to_dict())
                    
                    # 增强文本
                    if 'text' in row and pd.notna(row['text']):
                        augmented_texts = self.augment_text(row['text'], num_aug)
                        for text in augmented_texts:
                            new_row = row.to_dict()
                            new_row['text'] = text
                            new_row['augmented'] = 1
                            new_row['augment_type'] = 'text'
                            augmented_samples.append(new_row)
                    
                    # 增强URL
                    if 'url' in row and pd.notna(row['url']):
                        augmented_urls = self.augment_url(row['url'], num_aug)
                        for url in augmented_urls:
                            new_row = row.to_dict()
                            new_row['url'] = url
                            new_row['augmented'] = 1
                            new_row['augment_type'] = 'url'
                            augmented_samples.append(new_row)
                except Exception as e:
                    print(f"处理样本时出错: {e}")
                    continue
            
            augmented_df = pd.DataFrame(augmented_samples)
            
            # 去重
            if not augmented_df.empty:
                if "text" in augmented_df.columns and "url" in augmented_df.columns:
                    augmented_df = augmented_df.drop_duplicates(subset=["text", "url"], keep="last")
                elif "text" in augmented_df.columns:
                    augmented_df = augmented_df.drop_duplicates(subset=["text"], keep="last")
            
            print(f"增强后样本数量: {len(augmented_df)}")
            return augmented_df
        except Exception as e:
            print(f"增强数据集时出错: {e}")
            return df
    
    def process_all(self):
        """处理所有数据"""
        print("开始增强和优化数据集...")
        
        try:
            # 尝试加载版本数据集
            version_files = []
            if os.path.exists(f"{self.data_dir}/versions"):
                version_files = [f for f in os.listdir(f"{self.data_dir}/versions") if f.endswith('.csv')]
            
            if version_files:
                # 使用最新的版本数据集
                version_files.sort(reverse=True)
                latest_version = version_files[0]
                dataset_path = f"{self.data_dir}/versions/{latest_version}"
                print(f"使用最新版本数据集: {latest_version}")
                
                # 加载数据集
                df = pd.read_csv(dataset_path)
                
                # 划分训练集和验证集
                from sklearn.model_selection import train_test_split
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
            else:
                # 尝试加载其他数据集
                possible_paths = [
                    f"{self.data_dir}/multimodal/multimodal_total_dataset.csv",
                    f"{self.data_dir}/real/real_total_dataset.csv",
                    f"{self.data_dir}/synthetic/total_phishing_dataset.csv"
                ]
                
                dataset_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        dataset_path = path
                        break
                
                if dataset_path:
                    print(f"使用数据集: {dataset_path}")
                    df = pd.read_csv(dataset_path)
                    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
                else:
                    print("未找到数据集，使用模拟数据")
                    # 生成模拟数据
                    from data_preprocessing import DataPreprocessor
                    preprocessor = DataPreprocessor(self.data_dir)
                    df = preprocessor.build_synthetic_dataset(1000)
                    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
            
            # 处理样本不平衡
            train_df = self.handle_class_imbalance(train_df)
            val_df = self.handle_class_imbalance(val_df)
            
            # 增强训练集
            train_augmented = self.augment_dataset(train_df)
            
            # 保存增强后的数据集
            train_augmented.to_csv(f"{self.data_dir}/augmented/train_augmented.csv", index=False, encoding="utf-8-sig")
            val_df.to_csv(f"{self.data_dir}/augmented/val_augmented.csv", index=False, encoding="utf-8-sig")
            
            print("\n数据集增强和优化完成！")
            return train_augmented, val_df
        except Exception as e:
            print(f"处理数据时出错: {e}")
            return None, None

if __name__ == "__main__":
    augmenter = DataAugmenter()
    augmenter.process_all()
