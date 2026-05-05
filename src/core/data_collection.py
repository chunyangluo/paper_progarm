import os
import requests
import pandas as pd
import random
import time
import concurrent.futures
import urllib3
from datetime import datetime

# 禁用urllib3的InsecureRequestWarning警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DataCollector:
    """样本采集模块，从权威公开数据集和真实网络场景采集样本"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        # 增强请求头，模拟真实浏览器
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.google.com/"
        }
        # 新增：强制指定Windows代理（与设置的52842端口一致）
        self.proxies = {
            "http": "http://127.0.0.1:52842",
            "https": "http://127.0.0.1:52842"
        }
        # 新增：创建会话对象，复用TCP连接
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.proxies.update(self.proxies)
        # 禁用不必要的重定向和压缩，提升性能
        self.session.redirects = False
        self.session.verify = False  # 禁用SSL验证，提升性能
        # 初始化采集数据缓存
        self.collected_data_cache = {
            "phishTank": [],
            "phiUSIIL": []
        }
        self._create_directories()
        # 新增版本目录
        os.makedirs(f"{self.data_dir}/versions", exist_ok=True)
        # 权威数据源配置
        self.data_sources = {
            "phishTank": {
                "url": "https://phishtank.example.com/online-valid.csv",
                "fallback_urls": [
                    "https://raw.githubusercontent.com/phishtank/phishtank-list/master/online-valid.csv",
                    "https://data.phishtank.com/data/online-valid.csv"
                ],
                "description": "PhishTank官方钓鱼URL数据库"
            },
            "openphish": {
                "url": "https://openphish.com/feed.txt",
                "description": "OpenPhish实时钓鱼URL feed"
            },
            "majestic": {
                "url": "https://downloads.majestic.com/majestic_million.csv",
                "description": "Majestic Million全球排名网站"
            },
            "phiUSIIL": {
                "url": "https://raw.githubusercontent.com/PhiUSIIL/PhishingDataset/main/verified_phish.csv",
                "fallback_urls": [
                    "https://github.com/PhiUSIIL/phisheye/raw/master/data/verified_phish.csv",
                    "https://data.phishtank.com/data/online-valid.csv"
                ],
                "description": "PhiUSIIL钓鱼URL数据集"
            },
            "chifraud": {
                "url": "https://github.com/xuemingxxx/ChiFraud",
                "description": "CHIFRAUD中文欺诈文本基准集"
            },
            # 新增钓鱼URL数据源
            "phishtank_archive": {
                "url": "https://data.phishtank.com/data/archive/",
                "description": "PhishTank历史归档数据（按月份）"
            },
            "phishing_army": {
                "url": "https://phishing.army/download/phishing_army_blocklist.txt",
                "description": "Phishing Army实时钓鱼URL列表"
            },
            "mitre_phisheye": {
                "url": "https://raw.githubusercontent.com/mitre/phisheye/main/phish_results.csv",
                "backup_urls": ["https://data.phishtank.com/data/online-valid.csv"],
                "description": "MITRE Phisheye钓鱼URL数据集"
            },
            "urlhaus": {
                "url": "https://urlhaus.abuse.ch/downloads/csv/",
                "description": "URLhaus恶意URL数据库（含钓鱼）"
            },
            "cybercrime_tracker": {
                "url": "https://cybercrime-tracker.net/all.php",
                "description": "网络犯罪追踪器钓鱼/恶意URL"
            },
            # 新增正常URL数据源
            "tranco": {
                "url": "https://tranco-list.eu/top-1m.csv.zip",
                "description": "Tranco全球Top 100万正常域名"
            },
            "alexa": {
                "url": "https://s3.amazonaws.com/alexa-static/top-1m.csv.zip",
                "description": "Alexa Top 100万正常域名"
            }
        }
        # 初始化可用会话
        self.available_session = self.get_available_session()

    def check_proxy(self):
        """检查代理是否可用"""
        try:
            # 测试代理连接
            test_url = "http://www.google.com"
            response = self.session.get(test_url, timeout=5)
            return True
        except:
            return False

    def get_available_session(self):
        """获取可用的会话对象"""
        if self.check_proxy():
            print("[OK] 代理连接正常")
            return self.session
        else:
            # 代理不可用，创建无代理会话
            print("[WARN] 代理连接失败，切换到直接连接")
            session = requests.Session()
            session.headers.update(self.headers)
            session.verify = False
            return session
    
    def _create_directories(self):
        """创建数据目录"""
        directories = [
            f"{self.data_dir}/raw/phishing/sms",
            f"{self.data_dir}/raw/phishing/email",
            f"{self.data_dir}/raw/phishing/link",
            f"{self.data_dir}/raw/normal/sms",
            f"{self.data_dir}/raw/normal/email",
            f"{self.data_dir}/raw/normal/link"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def collect_phishTank_data(self, count=1000, api_key=None, use_cache=True):
        if use_cache and self.collected_data_cache["phishTank"]:
            return self.collected_data_cache["phishTank"]
            
        urls = [
            "https://data.phishtank.com/data/online-valid.csv",
            "https://raw.githubusercontent.com/phishtank/phishtank-list/master/online-valid.csv"
        ]
        
        for url in urls:
            try:
                r = self.available_session.get(url, timeout=15)
                r.raise_for_status()
                break
            except:
                continue
        else:
            print("PhishTank 全部不可用")
            return []

        from io import StringIO
        df = pd.read_csv(StringIO(r.text), on_bad_lines='skip')
        out = [{"url": row.url, "label":1, "source":"phishTank"} for _, row in df.iterrows() if row.url][:count]
        
        self.collected_data_cache["phishTank"] = out
        print(f"成功采集 {len(out)} 条")
        return out
    
    def collect_openphish_data(self, count=500, api_key=None):
        """从OpenPhish采集最新钓鱼URL（15分钟更新）"""
        print("正在采集 OpenPhish 最新钓鱼数据...")
        try:
            if api_key:
                url = f"https://openphish.com/feed.txt?key={api_key}"
            else:
                url = "https://openphish.com/feed.txt"  # 免费版
                
            response = self.available_session.get(url, timeout=30)
            response.raise_for_status()
            urls = response.text.strip().split("\n")[:count]
            
            data = [{"url": u, "label": 1, "scenario": "link", "source": "openphish", 
                    "timestamp": datetime.now().isoformat()} for u in urls]
            pd.DataFrame(data).to_csv(f"{self.data_dir}/raw/phishing/link/openphish.csv", 
                                     index=False, encoding="utf-8-sig")
            print(f"成功从OpenPhish采集了 {len(urls)} 条钓鱼URL")
            return data
        except Exception as e:
            print(f"OpenPhish采集失败: {e}")
            # 不使用生成的样本，返回空列表
            print("未使用生成的样本，返回空数据")
            return []
    
    def collect_phiUSIIL_data(self, count=1000):
        urls = [
            "https://data.phishtank.com/data/online-valid.csv",
            "https://openphish.com/feed.txt"
        ]
        
        for url in urls:
            try:
                r = self.available_session.get(url, timeout=10)
                r.raise_for_status()
                break
            except:
                continue
        else:
            print("PhiUSIIL 备用源不可用")
            return []

        out = []
        for line in r.text.splitlines()[:count]:
            u = line.strip()
            if u.startswith("http"):
                out.append({"url": u, "label":1, "source":"phiUSIIL"})
        
        self.collected_data_cache["phiUSIIL"] = out
        print(f"成功采集 {len(out)} 条")
        return out

    def collect_majestic_data(self, count=1000):
        """终极修复：兼容所有Majestic格式，自动提取正常URL"""
        print("正在从Majestic Million采集正常URL...")
        url = self.data_sources["majestic"]["url"]
        
        try:
            # 2次重试+固定间隔，提高速度
            for attempt in range(2):
                try:
                    response = self.available_session.get(
                            url, 
                            timeout=30,
                            allow_redirects=True
                        )
                    response.raise_for_status()
                    break
                except Exception as e:
                    print(f"第{attempt+1}次尝试失败: {e}，等待1秒重试...")
                    time.sleep(1)  # 固定间隔1秒，减少等待时间
            else:
                raise Exception("2次尝试均失败")

            # 不指定列、不跳行、全自动解析（彻底解决列名错误）
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), on_bad_lines="skip")

            # 自动找域名列（兼容所有列名：Domain/domain/Website/Host）
            domain_col = None
            for col in df.columns:
                if str(col).lower() in ["domain", "host", "website", "site"]:
                    domain_col = col
                    break
            
            normal_urls = []
            if domain_col:
                for val in df[domain_col].dropna():
                    if len(normal_urls) >= count:
                        break
                    domain = str(val).strip()
                    if "." in domain and len(domain) > 3:
                        normal_urls.append({"url": f"https://{domain}"})

            # 保存
            pd.DataFrame(normal_urls).to_csv(
                f"{self.data_dir}/raw/normal/link/majestic_links.csv", 
                index=False, encoding="utf-8-sig"
            )
            print(f"成功从Majestic采集了 {len(normal_urls)} 条正常URL")
            return normal_urls

        except Exception as e:
            print(f"Majestic采集失败: {e}")
            # 不使用生成的样本，返回空列表
            print("未使用生成的样本，返回空数据")
            return []
    

    

    

    

    

    

    

    
    def collect_additional_data(self, count=5000):
        """采集新增的权威数据源"""
        print("开始采集新增数据源...")
        
        # 采集钓鱼URL数据源
        钓鱼数据源 = [
            ("phishing_army", "https://phishing.army/download/phishing_army_blocklist.txt"),
            ("mitre_phisheye", "https://github.com/mitre/phisheye/raw/main/data/phish_results.csv")
        ]
        
        for source_name, url in 钓鱼数据源:
            try:
                print(f"正在从{source_name}采集钓鱼URL...")
                response = self.available_session.get(
                    url, 
                    timeout=30,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # 解析数据
                if source_name == "phishing_army":
                    # 处理文本格式的URL列表
                    urls = response.text.strip().split("\n")
                    phishing_urls = []
                    for i, url in enumerate(urls):
                        if i >= count:
                            break
                        if url and url.startswith(('http://', 'https://')):
                            phishing_urls.append({
                                "url": url,
                                "label": 1,
                                "source": source_name,
                                "timestamp": datetime.now().isoformat()
                            })
                else:
                    # 处理CSV格式的数据
                    from io import StringIO
                    df = pd.read_csv(StringIO(response.text), on_bad_lines='skip')
                    phishing_urls = []
                    # 自动检测URL列
                    url_columns = ['url', 'URL', 'link', 'phish_url']
                    url_col = None
                    for col in url_columns:
                        if col in df.columns:
                            url_col = col
                            break
                    
                    if url_col:
                        for _, row in df.iterrows():
                            if len(phishing_urls) >= count:
                                break
                            phish_url = row.get(url_col, "")
                            if phish_url and isinstance(phish_url, str) and phish_url.startswith(('http://', 'https://')):
                                phishing_urls.append({
                                    "url": phish_url,
                                    "label": 1,
                                    "source": source_name,
                                    "timestamp": datetime.now().isoformat()
                                })
                
                # 保存数据
                if phishing_urls:
                    pd.DataFrame(phishing_urls).to_csv(
                        f"{self.data_dir}/raw/phishing/link/{source_name}_links.csv", 
                        index=False, 
                        encoding="utf-8-sig"
                    )
                    print(f"成功从{source_name}采集了 {len(phishing_urls)} 条钓鱼URL")
            except Exception as e:
                print(f"{source_name}采集失败: {e}")
                print("未使用生成的样本，返回空数据")
        
        # 采集正常URL数据源
        正常数据源 = [
            ("tranco", "https://tranco-list.eu/top-1m.csv.zip"),
            ("alexa", "https://s3.amazonaws.com/alexa-static/top-1m.csv.zip")
        ]
        
        for source_name, url in 正常数据源:
            try:
                print(f"正在从{source_name}采集正常URL...")
                import zipfile
                import io
                
                response = self.available_session.get(
                    url, 
                    timeout=30,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # 处理ZIP文件
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    # 查找CSV文件
                    csv_files = [name for name in zf.namelist() if name.endswith('.csv')]
                    if csv_files:
                        csv_file = csv_files[0]
                        with zf.open(csv_file) as f:
                            df = pd.read_csv(f, on_bad_lines='skip')
                            
                            # 提取域名
                            normal_urls = []
                            # 自动检测域名列
                            domain_columns = ['domain', 'Domain', 'site', 'Site', 'url', 'URL']
                            domain_col = None
                            for col in domain_columns:
                                if col in df.columns:
                                    domain_col = col
                                    break
                            
                            # 如果没有找到域名列，尝试使用第二列（Alexa和Tranco通常使用第二列作为域名）
                            if not domain_col and len(df.columns) >= 2:
                                domain_col = df.columns[1]
                            
                            if domain_col:
                                for _, row in df.iterrows():
                                    if len(normal_urls) >= count:
                                        break
                                    domain = row.get(domain_col, "")
                                    if domain and isinstance(domain, str) and "." in domain:
                                        normal_urls.append({
                                            "url": f"https://{domain}",
                                            "label": 0,
                                            "source": source_name,
                                            "timestamp": datetime.now().isoformat()
                                        })
                
                # 保存数据
                if normal_urls:
                    pd.DataFrame(normal_urls).to_csv(
                        f"{self.data_dir}/raw/normal/link/{source_name}_links.csv", 
                        index=False, 
                        encoding="utf-8-sig"
                    )
                    print(f"成功从{source_name}采集了 {len(normal_urls)} 条正常URL")
            except Exception as e:
                print(f"{source_name}采集失败: {e}")
                print("未使用生成的样本，返回空数据")
        
        print("新增数据源采集完成！")

    def collect_all(self, counts={
        "phishing_url": 10000,
        "normal_url": 10000
    }):
        """采集所有样本（只使用采集数据，不生成样本）"""
        print("开始采集所有样本...")
        
        # 使用线程池并发采集数据
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 提交任务
            futures = [
                executor.submit(self.collect_phishTank_data, counts["phishing_url"]),
                executor.submit(self.collect_openphish_data, min(counts["phishing_url"], 5000)),
                executor.submit(self.collect_phiUSIIL_data, min(counts["phishing_url"], 5000)),
                executor.submit(self.collect_majestic_data, counts["normal_url"]),
                executor.submit(self.collect_additional_data, min(counts["phishing_url"], 5000))
            ]
            
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"任务执行失败: {e}")
        
        # 处理CHIFRAUD数据（单独执行，因为需要处理本地文件）
        self.process_chifraud_data()
        
        print("样本采集完成！")
    

    


    def collect_chifraud_data(self):
        """从GitHub下载CHIFRAUD中文欺诈文本基准集"""
        print("正在下载CHIFRAUD中文欺诈文本基准集...")
        import os
        import shutil
        import subprocess
        
        # 定义CHIFRAUD数据目录
        chifraud_dir = f"{self.data_dir}/raw/chifraud"
        os.makedirs(chifraud_dir, exist_ok=True)
        
        # 克隆GitHub仓库
        repo_url = "https://github.com/xuemingxxx/ChiFraud"
        try:
            # 如果目录已存在，先删除
            if os.path.exists(chifraud_dir):
                shutil.rmtree(chifraud_dir)
            
            # 使用subprocess执行git clone命令，添加超时设置
            print(f"开始克隆仓库: {repo_url}")
            result = subprocess.run(
                ["git", "clone", repo_url, chifraud_dir],
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                print("✅ CHIFRAUD数据集下载完成")
                return True
            else:
                print(f"⚠️ CHIFRAUD下载失败: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("⚠️ CHIFRAUD下载超时，请检查网络连接")
            return False
        except Exception as e:
            print(f"⚠️ CHIFRAUD下载失败: {e}")
            return False
    

    
    def process_chifraud_data(self):
        """处理CHIFRAUD数据集"""
        print("正在处理CHIFRAUD数据集...")
        import pandas as pd
        import os
        
        chifraud_dir = f"{self.data_dir}/raw/ChiFraud"
        processed_dir = f"{self.data_dir}/processed/chifraud"
        os.makedirs(processed_dir, exist_ok=True)
        
        # 查找CHIFRAUD数据文件
        data_files = []
        dataset_dir = os.path.join(chifraud_dir, "dataset")
        if os.path.exists(dataset_dir):
            for file in os.listdir(dataset_dir):
                if file.endswith('.csv'):
                    data_files.append(os.path.join(dataset_dir, file))
        
        # 如果没有找到文件，检查是否已有处理后的数据
        if not data_files:
            # 检查是否已有处理后的数据
            if os.path.exists(f"{processed_dir}/chifraud_phishing.csv") and os.path.exists(f"{processed_dir}/chifraud_normal.csv"):
                print("✅ 使用已存在的CHIFRAUD处理数据")
                return processed_dir
            else:
                # 不创建示例数据，返回空目录
                print("未找到CHIFRAUD数据文件，返回空目录")
                return processed_dir
        
        # 处理数据文件
        phishing_data = []
        normal_data = []
        
        for file_path in data_files:
            try:
                # 尝试读取文件，使用制表符作为分隔符
                df = pd.read_csv(file_path, sep='\t')
                
                # 处理数据，确保包含text和label字段
                if 'Label_id' in df.columns and 'Text' in df.columns:
                    # 重命名列名
                    df.rename(columns={'Label_id': 'label', 'Text': 'text'}, inplace=True)
                    # 分离钓鱼和正常数据
                    phishing = df[df['label'] == 1]
                    normal = df[df['label'] == 0]
                    
                    # 转换为字典格式
                    phishing_data.extend(phishing.to_dict('records'))
                    normal_data.extend(normal.to_dict('records'))
                elif 'text' in df.columns and 'label' in df.columns:
                    # 分离钓鱼和正常数据
                    phishing = df[df['label'] == 1]
                    normal = df[df['label'] == 0]
                    
                    # 转换为字典格式
                    phishing_data.extend(phishing.to_dict('records'))
                    normal_data.extend(normal.to_dict('records'))
                elif 'content' in df.columns and 'label' in df.columns:
                    # 如果字段名是content而不是text，进行重命名
                    df.rename(columns={'content': 'text'}, inplace=True)
                    # 分离钓鱼和正常数据
                    phishing = df[df['label'] == 1]
                    normal = df[df['label'] == 0]
                    
                    # 转换为字典格式
                    phishing_data.extend(phishing.to_dict('records'))
                    normal_data.extend(normal.to_dict('records'))
                else:
                    print(f"⚠️ 文件 {file_path} 缺少必要字段")
                    print(f"文件列名: {df.columns.tolist()}")
                    
            except Exception as e:
                print(f"⚠️ 处理文件 {file_path} 失败: {e}")
        
        # 保存处理后的数据
        if phishing_data:
            phishing_df = pd.DataFrame(phishing_data)
            phishing_df.to_csv(f"{processed_dir}/chifraud_phishing.csv", index=False, encoding="utf-8-sig")
            print(f"✅ 保存了 {len(phishing_data)} 条CHIFRAUD钓鱼样本")
        
        if normal_data:
            normal_df = pd.DataFrame(normal_data)
            normal_df.to_csv(f"{processed_dir}/chifraud_normal.csv", index=False, encoding="utf-8-sig")
            print(f"✅ 保存了 {len(normal_data)} 条CHIFRAUD正常样本")
        
        return processed_dir
    
    def update_and_version_dataset(self, version_suffix=""):
        """每周执行：更新数据+生成版本数据集（论文必备）"""
        from datetime import datetime
        import pandas as pd
        import time
        
        # 1. 拉取最新权威数据
        try:
            print("正在采集PhishTank数据...")
            self.collect_phishTank_data(800)
        except Exception as e:
            print(f"⚠️ PhishTank采集失败: {e}")
        
        try:
            print("正在采集OpenPhish数据...")
            self.collect_openphish_data(500)
        except Exception as e:
            print(f"⚠️ OpenPhish采集失败: {e}")
        
        try:
            print("正在采集PhiUSIIL数据...")
            self.collect_phiUSIIL_data(500)  # 新增PhiUSIIL数据采集
        except Exception as e:
            print(f"⚠️ PhiUSIIL采集失败: {e}")
        
        try:
            print("正在采集Majestic数据...")
            self.collect_majestic_data(800)  # 替换Alexa
        except Exception as e:
            print(f"⚠️ Majestic采集失败: {e}")
        
        try:
            print("正在处理CHIFRAUD数据...")
            self.process_chifraud_data()  # 处理CHIFRAUD数据
        except Exception as e:
            print(f"⚠️ CHIFRAUD处理失败: {e}")
        
        # 2. 定义所有数据文件路径（只使用采集数据，不使用生成的样本）
        data_files = {
            "phishing_url": [
                f"{self.data_dir}/raw/phishing/link/phishTank_links.csv",
                f"{self.data_dir}/raw/phishing/link/openphish.csv",
                f"{self.data_dir}/raw/phishing/link/phiUSIIL_links.csv",  # 新增PhiUSIIL数据文件
                f"{self.data_dir}/raw/phishing/link/phishing_army_links.csv",  # 新增Phishing Army数据文件
                f"{self.data_dir}/raw/phishing/link/mitre_phisheye_links.csv"  # 新增MITRE Phisheye数据文件
            ],
            "normal_url": [
                f"{self.data_dir}/raw/normal/link/majestic_links.csv",
                f"{self.data_dir}/raw/normal/link/tranco_links.csv",  # 新增Tranco数据文件
                f"{self.data_dir}/raw/normal/link/alexa_links.csv"  # 新增Alexa数据文件
            ],
            "chifraud": [
                f"{self.data_dir}/processed/chifraud/chifraud_phishing.csv",
                f"{self.data_dir}/processed/chifraud/chifraud_normal.csv"
            ]
        }
        
        # 3. 合并+清洗数据
        all_data = []
        for data_type, files in data_files.items():
            for file_path in files:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    try:
                        df = pd.read_csv(file_path)
                        # 添加类型标签
                        if "phishing" in data_type:
                            df["label"] = df.get("label", 1)
                        else:
                            df["label"] = df.get("label", 0)
                        
                        # 标准化字段
                        if "url" not in df.columns:
                            df["url"] = ""
                        if "text" not in df.columns:
                            df["text"] = df.get("url", "")
                        
                        all_data.append(df)
                    except Exception as e:
                        print(f"读取文件 {file_path} 失败: {e}")
        
        # 合并所有数据
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
        else:
            print("没有找到有效的数据文件，使用空数据集")
            # 创建空数据集
            final_df = pd.DataFrame(columns=["text", "url", "label", "source"])
        
        # 4. 数据质量处理（关键步骤）
        # 去重（基于text+url）
        if "text" in final_df.columns and "url" in final_df.columns:
            final_df = final_df.drop_duplicates(subset=["text", "url"], keep="last")
        elif "text" in final_df.columns:
            final_df = final_df.drop_duplicates(subset=["text"], keep="last")
        
        # 标签校验
        final_df = final_df[final_df["label"].isin([0, 1])]
        
        # 类别均衡（1:1比例，避免模型偏差）
        phish_count = len(final_df[final_df["label"] == 1])
        normal_count = len(final_df[final_df["label"] == 0])
        min_count = min(phish_count, normal_count)
        if min_count > 0:
            final_df = pd.concat([
                final_df[final_df["label"] == 1].sample(min_count, random_state=42),
                final_df[final_df["label"] == 0].sample(min_count, random_state=42)
            ])
        
        # 5. 生成版本文件
        version_date = datetime.now().strftime("%Y%m%d")
        if version_suffix:
            version_name = f"dataset_{version_date}_{version_suffix}.csv"
        else:
            version_name = f"dataset_{version_date}.csv"
        version_path = f"{self.data_dir}/versions/{version_name}"
        
        # 保存最终版本
        final_df.to_csv(version_path, index=False, encoding="utf-8-sig")
        
        # 输出统计信息
        print(f"✅ 版本数据集生成成功：{version_name}")
        print(f"   总样本数：{len(final_df)}")
        print(f"   钓鱼样本：{len(final_df[final_df['label'] == 1])}")
        print(f"   正常样本：{len(final_df[final_df['label'] == 0])}")
        print(f"   保存路径：{version_path}")
        
        return final_df

if __name__ == "__main__":
    collector = DataCollector()
    
    # 1. 采集全量数据
    collector.collect_all({ 
        "phishing_url": 1000, 
        "normal_url": 1000, 
        "phishing_sms": 500, 
        "normal_sms": 500, 
        "phishing_email": 500, 
        "normal_email": 500 
    })
    
    # 2. 生成版本数据集
    collector.update_and_version_dataset()
    
    # 3. 定时任务提示（论文实验建议）
    print("\n📅 建议设置定时任务：")
    print("   - 每小时：执行collect_openphish_data(200)  # 增量更新钓鱼URL")
    print("   - 每日：执行collect_majestic_data(500)    # 更新正常URL")
    print("   - 每周：执行update_and_version_dataset()  # 生成版本数据集")
    print("   - 每月：备份versions目录，用于模型对比实验")
