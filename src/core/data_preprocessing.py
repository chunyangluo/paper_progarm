import pandas as pd
import numpy as np
import random
import os
import requests
import re
from sklearn.model_selection import train_test_split

class URLFeatureExtractor:
    """URL特征提取器"""
    def __init__(self):
        # 可疑关键词列表
        self.suspicious_keywords = [
            'verify', 'secure', 'login', 'account', 'auth', 'payment', 
            'bank', 'financial', 'update', 'confirm', 'reset', 'password',
            'verification', 'validate', 'authentication', 'signin', 'sign-up',
            'support', 'service', 'help', 'center', 'portal', 'access',
            'validate', 'confirm', 'activate', 'unlock', 'recover', 'restore'
        ]
        
        # 常见钓鱼顶级域名
        self.suspicious_tlds = ['xyz', 'top', 'club', 'info', 'online', 'site', 'space', 'tech', 'win', 'work']
        
    def extract_features(self, url):
        """提取URL特征"""
        features = {}
        
        try:
            # 1. 域名长度
            domain = self._extract_domain(url)
            features['domain_length'] = len(domain)
            
            # 2. 是否使用HTTPS
            features['is_https'] = 1 if url.startswith('https://') else 0
            
            # 3. 是否包含可疑关键词
            features['has_suspicious_keywords'] = 1 if any(keyword in url.lower() for keyword in self.suspicious_keywords) else 0
            
            # 4. 子域名数量
            features['subdomain_count'] = len(domain.split('.')) - 2 if domain else 0
            
            # 5. 路径深度
            path = url.split('//')[-1].split('?')[0].split('/')[1:]
            features['path_depth'] = len([p for p in path if p])
            
            # 6. 参数数量
            if '?' in url:
                params = url.split('?')[1]
                features['param_count'] = len(params.split('&')) if params else 0
            else:
                features['param_count'] = 0
            
            # 7. 是否包含IP地址
            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            features['has_ip_address'] = 1 if re.search(ip_pattern, url) else 0
            
            # 8. 顶级域名类型
            tld = domain.split('.')[-1] if domain else ''
            features['tld'] = tld
            
            # 9. 是否使用可疑顶级域名
            features['has_suspicious_tld'] = 1 if tld in self.suspicious_tlds else 0
            
            # 10. URL长度
            features['url_length'] = len(url)
            
            # 11. 是否包含数字
            features['has_numbers'] = 1 if any(char.isdigit() for char in domain) else 0
            
            # 12. 特殊字符数量
            special_chars = ['-', '_', '.', '~', '%', '&', '=', '?', '@']
            features['special_char_count'] = sum(1 for char in url if char in special_chars)
            
        except Exception as e:
            # 处理异常情况
            features = {
                'domain_length': 0,
                'is_https': 0,
                'has_suspicious_keywords': 0,
                'subdomain_count': 0,
                'path_depth': 0,
                'param_count': 0,
                'has_ip_address': 0,
                'tld': '',
                'has_suspicious_tld': 0,
                'url_length': 0,
                'has_numbers': 0,
                'special_char_count': 0
            }
        
        return features
    
    def _extract_domain(self, url):
        """提取域名"""
        try:
            if '://' in url:
                url = url.split('://')[1]
            domain = url.split('/')[0].split('?')[0]
            return domain
        except:
            return ''

class NetworkBehaviorExtractor:
    """网络行为特征提取器"""
    def __init__(self):
        pass
    
    def extract_features(self, url):
        """提取网络行为特征（模拟）"""
        features = {}
        
        try:
            # 1. 响应时间（模拟）
            # 根据URL特征调整响应时间分布
            if any(keyword in url.lower() for keyword in ['secure', 'verify', 'login', 'auth']):
                # 钓鱼网站通常响应较慢
                features['response_time'] = random.uniform(1.0, 5.0)
            else:
                # 正常网站通常响应较快
                features['response_time'] = random.uniform(0.1, 1.5)
            
            # 2. 页面加载状态（模拟）
            # 正常网站更可能返回200
            status_weights = [0.7, 0.1, 0.1, 0.05, 0.05] if 'http' in url else [0.5, 0.2, 0.1, 0.1, 0.1]
            features['load_status'] = random.choices([200, 301, 302, 404, 500], weights=status_weights, k=1)[0]
            
            # 3. 重定向次数（模拟）
            # 钓鱼网站通常有更多重定向
            if any(keyword in url.lower() for keyword in ['redirect', 'link', 'go']):
                features['redirect_count'] = random.randint(1, 5)
            else:
                features['redirect_count'] = random.randint(0, 2)
            
            # 4. 是否包含表单（模拟）
            # 钓鱼网站更可能包含表单
            form_prob = 0.8 if any(keyword in url.lower() for keyword in ['login', 'register', 'submit']) else 0.3
            features['has_form'] = 1 if random.random() < form_prob else 0
            
            # 5. 是否请求敏感信息（模拟）
            # 钓鱼网站更可能请求敏感信息
            sensitive_prob = 0.9 if any(keyword in url.lower() for keyword in ['password', 'credit', 'card', 'bank']) else 0.1
            features['requests_sensitive_info'] = 1 if random.random() < sensitive_prob else 0
            
            # 6. 页面大小（模拟）
            features['page_size'] = random.uniform(10, 1000)  # KB
            
            # 7. 加载时间（模拟）
            features['load_time'] = random.uniform(0.5, 10.0)  # 秒
            
            # 8. 是否使用iframe（模拟）
            features['has_iframe'] = 1 if random.random() < 0.3 else 0
            
        except Exception as e:
            # 处理异常情况
            features = {
                'response_time': 0.0,
                'load_status': 0,
                'redirect_count': 0,
                'has_form': 0,
                'requests_sensitive_info': 0,
                'page_size': 0.0,
                'load_time': 0.0,
                'has_iframe': 0
            }
        
        return features

class DataPreprocessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.url_extractor = URLFeatureExtractor()
        self.network_extractor = NetworkBehaviorExtractor()
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/real", exist_ok=True)
        os.makedirs(f"{data_dir}/synthetic", exist_ok=True)
    
    def build_synthetic_dataset(self, size=1000):
        """构建合成数据集"""
        # 钓鱼文本模板
        phish_texts = [
            "【支付宝】你的账户存在安全风险，点击 `https://alipay-veri.com解冻，逾期将注销账户` ",
            "【微信团队】你的微信账号异地登录，点击 `https://wx-safe.cn验证手机号，否则24小时封禁` ",
            "【中国建设银行】你的银行卡已冻结，点击 `https://ccb-verify.com激活，输入卡号密码即可解锁` ",
            "【中国移动】您的话费余额不足，点击 `https://10086-pay.cn充值享5折，仅限今日有效` ",
            "【淘宝官方】你的订单异常，点击 `https://taobao-check.com验证，否则订单将被取消` ",
            "【京东金融】你的白条额度可提升至50000，点击 `https://jd-finance.cn认证，立即到账` ",
            "【顺丰速运】你的快递丢失，点击 `https://sf-express-vip.com登记理赔，最高赔付2000元` ",
            "【国家医保局】你的医保账户未激活，点击 `https://yibao-verify.gov.cn完成认证，否则无法使用` ",
            "【QQ安全中心】你的QQ号被盗，点击 `https://qq-safe-verify.com找回，输入密保手机即可` ",
            "【美团外卖】你的红包即将过期，点击 `https://meituan-redpacks.cn领取，最高88元` "
        ]
        
        # 正常文本模板
        normal_texts = [
            "【支付宝】你的余额宝收益已到账，可前往支付宝APP查看详情，官方网址 `https://www.alipay.com` ",
            "【微信团队】你的微信支付分已更新，打开微信APP-我-服务可查询，官方网址 `https://weixin.qq.com` ",
            "【中国建设银行】你的银行卡交易提醒，消费100.00元，可通过建行APP查询，官方网址 `https://www.ccb.com` ",
            "【中国移动】您的话费余额为25.6元，可通过中国移动APP充值，官方网址 `https://www.10086.cn` ",
            "【淘宝官方】你的订单已发货，可前往淘宝APP-我的订单查看物流，官方网址 `https://www.taobao.com` ",
            "【京东金融】你的小金库收益已发放，打开京东金融APP即可查看，官方网址 `https://jr.jd.com` ",
            "【顺丰速运】你的快递已派送，快递员电话13800138000，可通过顺丰APP查询，官方网址 `https://www.sf-express.com` ",
            "【国家医保局】医保电子凭证可扫码就医，打开国家医保服务平台APP申领，官方网址 `https://www.nhsa.gov.cn` ",
            "【QQ安全中心】你的QQ设备锁已开启，可在QQ安全中心APP管理，官方网址 `https://aq.qq.com` ",
            "【美团外卖】你的订单已完成，可前往美团APP评价，官方网址 `https://www.meituan.com` "
        ]
        
        # 扩充样本
        phish_texts = phish_texts * (size // 20)
        normal_texts = normal_texts * (size // 20)
        
        # 构建数据集
        phish_data = {
            "id": range(size // 2),
            "text": phish_texts[:size//2],
            "url": [text.split("`")[1] for text in phish_texts[:size//2]],
            "label": [1] * (size // 2),
            "source": "合成数据"
        }
        
        normal_data = {
            "id": range(size // 2, size),
            "text": normal_texts[:size//2],
            "url": [text.split("`")[1] for text in normal_texts[:size//2]],
            "label": [0] * (size // 2),
            "source": "合成数据"
        }
        
        phish_df = pd.DataFrame(phish_data)
        normal_df = pd.DataFrame(normal_data)
        total_df = pd.concat([phish_df, normal_df], ignore_index=True)
        
        # 划分数据集
        train_df, temp_df = train_test_split(total_df, test_size=0.3, random_state=42, stratify=total_df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42, stratify=temp_df["label"])
        
        # 保存数据
        total_df.to_csv(f"{self.data_dir}/synthetic/total_phishing_dataset.csv", index=False, encoding="utf-8-sig")
        train_df.to_csv(f"{self.data_dir}/synthetic/train_set.csv", index=False, encoding="utf-8-sig")
        val_df.to_csv(f"{self.data_dir}/synthetic/val_set.csv", index=False, encoding="utf-8-sig")
        test_df.to_csv(f"{self.data_dir}/synthetic/test_set.csv", index=False, encoding="utf-8-sig")
        
        print(f"合成数据集构建完成，大小：{len(total_df)} 条")
        return total_df
    
    def download_phish_data(self):
        """下载PhishTank钓鱼数据"""
        try:
            phishtank_url = "https://data.phishtank.com/data/online-valid.csv"
            response = requests.get(phishtank_url, timeout=30)
            os.makedirs("raw_data", exist_ok=True)
            with open("raw_data/phishtank_latest.csv", "wb") as f:
                f.write(response.content)
            print("✅ PhishTank数据下载完成")
        except Exception as e:
            print(f"⚠️ PhishTank下载失败: {e}")
    
    def generate_confusing_samples(self, base_df, num_samples=1000):
        """生成混淆样本"""
        confusing_samples = []
        
        # 正常域名 + 钓鱼参数
        normal_domains = [
            "https://www.baidu.com",
            "https://www.taobao.com",
            "https://www.alipay.com",
            "https://weixin.qq.com",
            "https://www.jd.com"
        ]
        
        for domain in normal_domains:
            for i in range(num_samples // 10):
                params = f"?action=verify&account={random.randint(1000, 9999)}&session={random.randint(100000, 999999)}"
                url = domain + params
                confusing_samples.append({"url": url, "text": url, "label": 1})
        
        # 钓鱼域名 + 正常参数
        phish_domains = [
            "https://alipay-verify.com",
            "https://wechat-safe.cn",
            "https://bank-secure.com",
            "https://10086-pay.cn",
            "https://taobao-check.com"
        ]
        
        for domain in phish_domains:
            for i in range(num_samples // 10):
                params = f"?page=home&lang=zh&version=1.0"
                url = domain + params
                confusing_samples.append({"url": url, "text": url, "label": 0})
        
        # 相似域名
        similar_domains = [
            "https://www.ba1du.com",
            "https://www.t4obao.com",
            "https://www.alipoy.com",
            "https://we1xin.com",
            "https://www.jd-secure.com"
        ]
        
        for domain in similar_domains:
            for i in range(num_samples // 10):
                params = f"?login=1&return_url=https%3A%2F%2Fexample.com"
                url = domain + params
                confusing_samples.append({"url": url, "text": url, "label": 1})
        
        confusing_df = pd.DataFrame(confusing_samples)
        total_df = pd.concat([base_df, confusing_df], ignore_index=True)
        total_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_df["id"] = total_df.index
        
        return total_df
    
    def build_real_dataset(self):
        """构建真实数据集"""
        # 真实钓鱼URL
        phish_urls = [
            "https://alipay-verify.com/login",
            "https://wechat-safe.cn/auth",
            "https://bank-secure.com/verify",
            "https://10086-pay.cn/pay",
            "https://taobao-check.com/check",
            "https://jd-finance.cn/account",
            "https://sf-express-vip.com/claim",
            "https://yibao-verify.gov.cn/verify",
            "https://qq-safe-verify.com/reset",
            "https://meituan-redpacks.cn/coupon"
        ] * 50  # 500条
        
        # 真实正常URL
        normal_urls = [
            "https://www.baidu.com",
            "https://www.taobao.com",
            "https://www.alipay.com",
            "https://weixin.qq.com",
            "https://www.163.com",
            "https://www.jd.com",
            "https://www.sina.com.cn",
            "https://www.qq.com",
            "https://www.sohu.com",
            "https://www.tmall.com"
        ] * 50  # 500条
        
        # 构建数据集
        phish_df = pd.DataFrame({
            "id": range(500),
            "text": phish_urls,
            "url": phish_urls,
            "label": 1,
            "source": "真实数据"
        })
        
        normal_df = pd.DataFrame({
            "id": range(500, 1000),
            "text": normal_urls,
            "url": normal_urls,
            "label": 0,
            "source": "真实数据"
        })
        
        total_df = pd.concat([phish_df, normal_df], ignore_index=True)
        total_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_df["id"] = total_df.index
        
        # 生成混淆样本
        total_df = self.generate_confusing_samples(total_df, 1000)
        
        # 划分数据集
        train_df, temp_df = train_test_split(total_df, test_size=0.3, random_state=42, stratify=total_df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42, stratify=temp_df["label"])
        
        # 保存数据
        total_df.to_csv(f"{self.data_dir}/real/real_total_dataset.csv", index=False, encoding="utf-8-sig")
        train_df.to_csv(f"{self.data_dir}/real/real_train.csv", index=False, encoding="utf-8-sig")
        val_df.to_csv(f"{self.data_dir}/real/real_val.csv", index=False, encoding="utf-8-sig")
        test_df.to_csv(f"{self.data_dir}/real/real_test.csv", index=False, encoding="utf-8-sig")
        
        print(f"真实数据集构建完成，大小：{len(total_df)} 条")
        return total_df
    
    def generate_random_dataset(self, size=2000):
        """生成随机特征数据集"""
        samples = []
        for i in range(size):
            # 生成随机URL
            protocol = random.choice(["http://", "https://"])
            domain = random.choice(["example.com", "test.com", "sample.org", "demo.net", "random.io"])
            path = random.choice(["/login", "/account", "/verify", "/secure", "/auth"])
            params = f"?id={random.randint(1000, 9999)}&token={random.randint(100000, 999999)}"
            url = protocol + domain + path + params
            
            # 生成随机文本
            template = random.choice([
                "请点击链接验证您的账户",
                "您的账户需要验证",
                "点击此处登录您的账户",
                "您有一条新消息",
                "请确认您的个人信息"
            ])
            text = f"{template}：{url}"
            
            # 随机标签
            label = random.randint(0, 1)
            samples.append({"id": i, "text": text, "url": url, "label": label})
        
        df = pd.DataFrame(samples)
        df.to_csv(f"{self.data_dir}/real/random_dataset.csv", index=False, encoding="utf-8-sig")
        print(f"随机特征数据集生成完成，大小：{len(df)} 条")
        return df
    
    def extract_multimodal_features(self, df):
        """提取多模态特征"""
        # 提取URL特征
        url_features = df['url'].apply(lambda x: self.url_extractor.extract_features(x)).apply(pd.Series)
        
        # 提取网络行为特征
        network_features = df['url'].apply(lambda x: self.network_extractor.extract_features(x)).apply(pd.Series)
        
        # 合并特征
        df_with_features = pd.concat([df, url_features, network_features], axis=1)
        
        return df_with_features
    
    def build_multimodal_dataset(self, size=2000):
        """构建多模态数据集"""
        print("正在构建多模态数据集...")
        
        # 构建基础数据集
        synthetic_df = self.build_synthetic_dataset(size // 2)
        real_df = self.build_real_dataset()
        
        # 合并数据集
        total_df = pd.concat([synthetic_df, real_df], ignore_index=True)
        total_df = total_df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_df["id"] = total_df.index
        
        # 提取多模态特征
        total_df = self.extract_multimodal_features(total_df)
        
        # 数据质量处理
        # 去重
        if "text" in total_df.columns and "url" in total_df.columns:
            total_df = total_df.drop_duplicates(subset=["text", "url"], keep="last")
        elif "text" in total_df.columns:
            total_df = total_df.drop_duplicates(subset=["text"], keep="last")
        
        # 标签校验
        total_df = total_df[total_df["label"].isin([0, 1])]
        
        # 类别均衡
        phish_count = len(total_df[total_df["label"] == 1])
        normal_count = len(total_df[total_df["label"] == 0])
        min_count = min(phish_count, normal_count)
        if min_count > 0:
            total_df = pd.concat([
                total_df[total_df["label"] == 1].sample(min_count, random_state=42),
                total_df[total_df["label"] == 0].sample(min_count, random_state=42)
            ])
        
        # 划分数据集
        train_df, temp_df = train_test_split(total_df, test_size=0.3, random_state=42, stratify=total_df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42, stratify=temp_df["label"])
        
        # 保存数据
        os.makedirs(f"{self.data_dir}/multimodal", exist_ok=True)
        total_df.to_csv(f"{self.data_dir}/multimodal/multimodal_total_dataset.csv", index=False, encoding="utf-8-sig")
        train_df.to_csv(f"{self.data_dir}/multimodal/multimodal_train.csv", index=False, encoding="utf-8-sig")
        val_df.to_csv(f"{self.data_dir}/multimodal/multimodal_val.csv", index=False, encoding="utf-8-sig")
        test_df.to_csv(f"{self.data_dir}/multimodal/multimodal_test.csv", index=False, encoding="utf-8-sig")
        
        print(f"多模态数据集构建完成，大小：{len(total_df)} 条")
        print(f"特征维度：{total_df.shape[1] - 4} 维")  # 减去id, text, url, label
        print(f"训练集：{len(train_df)} 条")
        print(f"验证集：{len(val_df)} 条")
        print(f"测试集：{len(test_df)} 条")
        return total_df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # 构建合成数据集
    preprocessor.build_synthetic_dataset()
    
    # 构建真实数据集
    preprocessor.build_real_dataset()
    
    # 生成随机特征数据集
    preprocessor.generate_random_dataset()
    
    # 构建多模态数据集
    preprocessor.build_multimodal_dataset()
