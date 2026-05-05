import os
import sys
import json
import time
import random
import hashlib
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'versions')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

EXISTING_COLUMNS = ['url', 'phish_id', 'submission_time', 'verification_time',
                    'online', 'target', 'label', 'source', 'text', 'scenario',
                    'timestamp', 'rank']


class SupplementaryDataCollector:

    def __init__(self):
        self.collected = []
        self.stats = {
            'phishing_sms': 0, 'phishing_email': 0, 'phishing_link': 0,
            'normal_sms': 0, 'normal_email': 0, 'normal_general': 0,
            'total_phishing': 0, 'total_normal': 0
        }

    def _add_sample(self, text, label, scenario, source, url='', **kwargs):
        sample = {
            'url': url,
            'phish_id': kwargs.get('phish_id', ''),
            'submission_time': kwargs.get('submission_time', ''),
            'verification_time': kwargs.get('verification_time', ''),
            'online': kwargs.get('online', ''),
            'target': kwargs.get('target', ''),
            'label': label,
            'source': source,
            'text': text,
            'scenario': scenario,
            'timestamp': datetime.now().isoformat(),
            'rank': kwargs.get('rank', ''),
        }
        self.collected.append(sample)
        if label == 1:
            self.stats['total_phishing'] += 1
            if scenario == 'sms':
                self.stats['phishing_sms'] += 1
            elif scenario == 'email':
                self.stats['phishing_email'] += 1
            elif scenario == 'link':
                self.stats['phishing_link'] += 1
        else:
            self.stats['total_normal'] += 1
            if scenario == 'sms':
                self.stats['normal_sms'] += 1
            elif scenario == 'email':
                self.stats['normal_email'] += 1
            else:
                self.stats['normal_general'] += 1

    def collect_phishing_sms(self, target=1500):
        logger.info(f"Collecting {target} phishing SMS samples...")
        templates = [
            "【{platform}】您的{account}存在安全风险，请立即点击 {url} 进行验证，否则将在24小时内{consequence}",
            "【{platform}】紧急通知：您的{account}已被冻结，请点击 {url} 解冻，逾期将{consequence}",
            "【{platform}】验证码：您的{account}正在异地登录，如非本人操作请点击 {url} 确认",
            "【{platform}】您的{account}异常，请立即登录 {url} 核实，否则将{consequence}",
            "【{platform}】尊敬的用户，您的{account}已过期，请点击 {url} 续期，逾期将{consequence}",
            "【{platform}】系统检测到您的{account}存在{risk}，请立即点击 {url} 处理",
            "【{platform}】您的{account}余额不足，请点击 {url} 充值，否则将{consequence}",
            "【{platform}】恭喜您获得{reward}，请点击 {url} 领取，仅限今日！",
            "【{platform}】您的{account}已被限制，请点击 {url} 申请解封",
            "【{platform}】重要提醒：您的{account}即将{action}，请点击 {url} 确认保留",
        ]
        platforms = ['支付宝', '微信', '银行', '工商银行', '建设银行', '农业银行', '中国银行',
                     '招商银行', '交通银行', '邮储银行', '京东', '淘宝', '拼多多', '美团',
                     '抖音', '快手', '微博', 'QQ', '12306', '中国移动', '中国联通', '中国电信',
                     '顺丰快递', '京东快递', '中通快递', '韵达快递', '社保局', '税务局', '公积金中心']
        accounts = ['账户', '账号', '支付账户', '银行卡', '信用卡', '会员', '积分账户', '社保账户',
                    '公积金账户', '手机号', '邮箱', '身份信息', '信用额度']
        consequences = ['永久冻结', '自动注销', '影响征信', '无法恢复', '被永久封禁',
                       '限制使用', '降低信用分', '产生滞纳金']
        risks = ['异常登录', '安全风险', '被盗风险', '信息泄露', '违规操作']
        rewards = ['iPhone16大奖', '现金红包888元', '免费旅游券', '购物免单资格',
                  'VIP会员', '100元话费', '50元优惠券', '限量版礼品']
        actions = ['过期', '被注销', '被降级', '被清零', '失效']
        domains = ['verify', 'secure', 'auth', 'login', 'confirm', 'update', 'check',
                  'safe', 'protect', 'reset', 'unlock', 'service', 'help', 'support']
        suffixes = ['.cn', '.com', '.net', '.xyz', '.top', '.cc', '.vip']

        count = 0
        while count < target:
            template = random.choice(templates)
            platform = random.choice(platforms)
            account = random.choice(accounts)
            consequence = random.choice(consequences)
            risk = random.choice(risks)
            reward = random.choice(rewards)
            action = random.choice(actions)
            domain = random.choice(domains)
            suffix = random.choice(suffixes)
            url = f"https://{platform.lower()}-{domain}{suffix}/{hashlib.md5(str(random.random()).encode()).hexdigest()[:6]}"

            text = template.format(
                platform=platform, account=account, consequence=consequence,
                risk=risk, reward=reward, action=action, url=url
            )
            self._add_sample(text, 1, 'sms', 'synthetic_sms', url=url)
            count += 1
        logger.info(f"  Collected {count} phishing SMS samples")

    def collect_phishing_email(self, target=1500):
        logger.info(f"Collecting {target} phishing email samples...")
        templates = [
            "尊敬的{title}，\n\n您的{account}存在{issue}，请立即登录 {url} 进行{action}。\n\n如非本人操作，请忽略此邮件。但请注意，如果您不在{deadline}内完成验证，您的{account}将被{consequence}。\n\n此致\n{platform}安全中心",
            "亲爱的{title}，\n\n我们检测到您的{account}于{time}在{location}进行了异常操作。为保障您的资金安全，请立即点击 {url} 确认身份。\n\n如非本人操作，请立即修改密码并联系客服。\n\n{platform}安全团队",
            "尊敬的{title}，\n\n您的{account}已通过{event}审核，请点击 {url} 领取{reward}。\n\n请在{deadline}前完成领取，逾期将视为自动放弃。\n\n{platform}运营中心",
            "尊敬的{title}，\n\n根据{authority}通知，您的{account}存在{issue}，需要您在{deadline}前登录 {url} 完成{action}。\n\n逾期未处理将{consequence}。\n\n{authority}通知中心",
            "亲爱的用户，\n\n您的{account}已被临时{status}，原因是{reason}。如需恢复使用，请点击 {url} 提交申诉材料。\n\n{platform}客服中心",
        ]
        titles = ['用户', '客户', '会员', '先生/女士', '先生', '女士']
        accounts = ['银行账户', '信用卡', '支付宝账户', '微信支付', '网银账户', '社保账户',
                    '公积金账户', '投资账户', '贷款账户', '电子钱包']
        issues = ['安全风险', '异常交易', '身份验证过期', '信息不完整', '违规操作',
                 '未授权访问', '密码泄露风险', '账户被盗']
        actions = ['身份验证', '安全确认', '信息更新', '密码重置', '账户解冻', '风险排查']
        deadlines = ['24小时', '48小时', '3个工作日', '7天', '本月底']
        consequences = ['永久冻结', '自动注销', '限制使用', '影响征信', '产生滞纳金']
        platforms = ['中国银行', '工商银行', '建设银行', '支付宝', '微信支付', '京东金融',
                    '招商银行', '交通银行', '银联', '蚂蚁金服']
        events = ['年度', '季度', '特别', '限时', '周年庆']
        rewards = ['现金奖励', '积分奖励', '优惠券', '免息额度', 'VIP权益']
        authorities = ['银保监会', '人民银行', '网信办', '公安部', '税务局']
        statuses = ['冻结', '限制', '锁定', '暂停']
        reasons = ['异常登录', '可疑交易', '信息变更', '安全审查', '系统升级']
        locations = ['异地', '未知设备', '境外', '非常用设备', '新设备']

        count = 0
        while count < target:
            template = random.choice(templates)
            domain = random.choice(['verify', 'secure', 'auth', 'service'])
            suffix = random.choice(['.cn', '.com', '.net'])
            url = f"https://{domain}-bank{suffix}/{hashlib.md5(str(random.random()).encode()).hexdigest()[:8]}"

            text = template.format(
                title=random.choice(titles),
                account=random.choice(accounts),
                issue=random.choice(issues),
                action=random.choice(actions),
                deadline=random.choice(deadlines),
                consequence=random.choice(consequences),
                platform=random.choice(platforms),
                url=url,
                time=f"{random.randint(1,28)}日{random.randint(0,23)}:{random.randint(0,59):02d}",
                location=random.choice(locations),
                event=random.choice(events),
                reward=random.choice(rewards),
                authority=random.choice(authorities),
                status=random.choice(statuses),
                reason=random.choice(reasons),
            )
            self._add_sample(text, 1, 'email', 'synthetic_email', url=url)
            count += 1
        logger.info(f"  Collected {count} phishing email samples")

    def collect_phishing_link(self, target=1000):
        logger.info(f"Collecting {target} phishing link samples...")
        templates = [
            "点击链接领取{reward} {url}",
            "限时优惠！{discount}折起，点击抢购 {url}",
            "您的{account}需要验证，点击链接 {url}",
            "免费领取{item}，仅限今日 {url}",
            "恭喜中奖！{reward}等你来领 {url}",
            "紧急！{platform}系统升级，请点击链接确认 {url}",
            "{platform}年度大促，{discount}折封顶 {url}",
            "您的订单{order_id}需要确认收货，点击 {url}",
            "邀请您加入{platform}VIP，享专属权益 {url}",
            "您的{account}即将到期，点击续费 {url}",
        ]
        rewards = ['888元红包', '100元话费', '免费会员', '50元优惠券', 'iPhone16']
        discounts = ['1', '2', '3', '5', '0.5']
        accounts = ['账户', '会员', '积分', '优惠券', '余额']
        items = ['iPhone16', '现金红包', '购物卡', '话费', 'VIP会员']
        platforms = ['淘宝', '京东', '拼多多', '美团', '抖音', '微信', '支付宝']
        order_ids = [f"ORD{random.randint(100000,999999)}" for _ in range(10)]

        count = 0
        while count < target:
            template = random.choice(templates)
            domain_words = ['free', 'gift', 'reward', 'claim', 'offer', 'deal', 'promo', 'sale']
            domain = random.choice(domain_words)
            suffix = random.choice(['.xyz', '.top', '.cc', '.vip', '.tk', '.ml', '.ga'])
            url = f"http://{domain}-shop{suffix}/{hashlib.md5(str(random.random()).encode()).hexdigest()[:6]}"

            text = template.format(
                reward=random.choice(rewards),
                url=url,
                discount=random.choice(discounts),
                account=random.choice(accounts),
                item=random.choice(items),
                platform=random.choice(platforms),
                order_id=random.choice(order_ids),
            )
            self._add_sample(text, 1, 'link', 'synthetic_link', url=url)
            count += 1
        logger.info(f"  Collected {count} phishing link samples")

    def collect_normal_sms(self, target=1500):
        logger.info(f"Collecting {target} normal SMS samples...")
        templates = [
            "【{platform}】您的验证码是{code}，{minutes}分钟内有效。请勿泄露给他人。",
            "【{platform}】您预约的{service}已确认，时间：{time}，地点：{location}。",
            "【{platform}】您的{order}已发货，快递单号：{tracking}，请注意查收。",
            "【{platform}】您本月账单金额{amount}元，还款日{date}，请按时还款。",
            "【{platform}】您的{service}办理成功，有效期至{date}。",
            "【{platform}】尊敬的会员，您有{points}积分即将过期，请及时使用。",
            "【{platform}】您的{service}预约已取消，如需重新预约请登录APP。",
            "【{platform}】感谢您的反馈，我们已收到您的{type}，将在{days}个工作日内回复。",
            "【{platform}】温馨提醒：您预约的{service}将在明天{time}开始，请准时参加。",
            "【{platform}】您的{service}申请已通过，详情请登录APP查看。",
            "【{platform}】您的快递已签收，感谢使用{platform}服务。",
            "【{platform}】系统维护通知：{date}凌晨2:00-6:00将进行系统升级，届时部分功能暂不可用。",
        ]
        platforms = ['中国移动', '中国联通', '中国电信', '顺丰', '京东', '淘宝', '美团',
                    '招商银行', '工商银行', '支付宝', '微信', '12306', '滴滴', '饿了么']
        services = ['体检', '面试', '会议', '培训', '咨询', '维修', '安装', '保养']
        locations = ['北京市朝阳区XX路XX号', '上海市浦东新区XX大道XX号',
                    '广州市天河区XX街XX号', '深圳市南山区XX路XX号']

        count = 0
        while count < target:
            template = random.choice(templates)
            text = template.format(
                platform=random.choice(platforms),
                code=f"{random.randint(100000, 999999)}",
                minutes=random.choice(['5', '10', '15', '30']),
                service=random.choice(services),
                time=f"{random.randint(8,20)}:{random.randint(0,59):02d}",
                location=random.choice(locations),
                order=f"ORD{random.randint(100000,999999)}",
                tracking=f"SF{random.randint(1000000000,9999999999)}",
                amount=f"{random.randint(10,5000)}.{random.randint(0,99):02d}",
                date=f"{random.randint(1,28)}日",
                points=f"{random.randint(100,5000)}",
                type=random.choice(['建议', '投诉', '咨询', '申请']),
                days=random.choice(['1', '3', '5', '7']),
            )
            self._add_sample(text, 0, 'sms', 'synthetic_normal_sms')
            count += 1
        logger.info(f"  Collected {count} normal SMS samples")

    def collect_normal_email(self, target=1000):
        logger.info(f"Collecting {target} normal email samples...")
        templates = [
            "尊敬的{title}，\n\n感谢您注册{platform}。您的账户已成功创建，请使用注册邮箱登录。\n\n如有任何问题，请联系客服。\n\n{platform}团队",
            "亲爱的{title}，\n\n您在{platform}的订单已确认。预计{days}个工作日内送达。\n\n订单详情请登录查看。\n\n{platform}客服",
            "尊敬的{title}，\n\n您的{service}申请已受理，预计{days}个工作日内完成审核。\n\n审核结果将通过邮件通知您。\n\n{platform}",
            "亲爱的{title}，\n\n感谢您参加{event}。活动回顾和资料已上传，请登录查看。\n\n期待下次再见！\n\n{platform}活动组",
            "尊敬的{title}，\n\n您的{platform}会员即将于{date}到期。如需续费，请登录官方APP操作。\n\n{platform}会员中心",
            "亲爱的{title}，\n\n您订阅的{newsletter}已更新。本期主题：{topic}。\n\n点击查看详情（仅限APP内）。\n\n{platform}编辑部",
        ]
        titles = ['用户', '客户', '会员', '先生/女士']
        platforms = ['京东', '淘宝', '美团', '携程', '知乎', 'B站', '豆瓣', '网易云音乐']
        services = ['退款', '换货', '发票', '会员升级', '密码重置']
        events = ['技术沙龙', '线上讲座', '读书会', '产品发布会', '年会']
        newsletters = ['周报', '月刊', '技术周刊', '生活杂志']
        topics = ['人工智能前沿', '健康生活指南', '职场发展', '科技趋势']

        count = 0
        while count < target:
            template = random.choice(templates)
            text = template.format(
                title=random.choice(titles),
                platform=random.choice(platforms),
                days=random.choice(['3', '5', '7', '10']),
                service=random.choice(services),
                event=random.choice(events),
                date=f"{random.randint(1,28)}日",
                newsletter=random.choice(newsletters),
                topic=random.choice(topics),
            )
            self._add_sample(text, 0, 'email', 'synthetic_normal_email')
            count += 1
        logger.info(f"  Collected {count} normal normal email samples")

    def collect_normal_general(self, target=1500):
        logger.info(f"Collecting {target} normal general samples...")
        templates = [
            "明天下午3点在会议室开项目进度汇报会，请准时参加。",
            "您好，我想咨询一下贵公司的产品价格和配送方式。",
            "周末一起去爬山吧，天气预报说周六是晴天。",
            "请问图书馆的开放时间是几点到几点？",
            "感谢您的来信，我们会在三个工作日内回复您。",
            "今天食堂的午餐有红烧肉和清炒时蔬，味道不错。",
            "下周一需要提交季度报告，请各位提前准备。",
            "您好，我想预约下周三上午的门诊。",
            "最近天气变化大，注意增减衣物，保重身体。",
            "这个周末有什么好看的电影推荐吗？",
            "请问这个商品还有其他颜色可选吗？",
            "我已收到您发送的文件，谢谢！",
            "会议纪要已整理完毕，请各位查阅并确认。",
            "您好，请问贵司的办公地址在哪里？我想上门咨询。",
            "今天的培训内容很有收获，感谢老师的分享。",
            "请问附近有停车场吗？收费标准是怎样的？",
            "我需要修改一下收货地址，请问怎么操作？",
            "您好，请问你们支持哪些支付方式？",
            "这个项目的截止日期是什么时候？",
            "请问你们提供售后服务吗？保修期多长？",
            "我想了解一下你们的产品功能，可以发一份介绍吗？",
            "今天的会议改到下午2点了，请注意调整安排。",
            "请问这个价格含税吗？可以开增值税发票吗？",
            "您好，我想了解一下会员权益和收费标准。",
            "请问你们的工作时间是怎样的？周末上班吗？",
            "我已完成了在线测评，请问多久可以出结果？",
            "请问附近有什么好吃的餐厅推荐？",
            "这个方案我觉得还需要再讨论一下，约个时间碰面吧。",
            "您好，请问可以提供产品的技术参数吗？",
            "最近在读一本关于人工智能的书，写得很好，推荐给大家。",
        ]

        count = 0
        while count < target:
            text = random.choice(templates)
            variation = random.choice([0, 1, 2])
            if variation == 1:
                text = text.replace("。", "！")
            elif variation == 2:
                text = text + " 期待您的回复。"
            self._add_sample(text, 0, 'general', 'synthetic_normal_general')
            count += 1
        logger.info(f"  Collected {count} normal general samples")

    def collect_all(self):
        logger.info("=" * 60)
        logger.info("  Supplementary Data Collection")
        logger.info("=" * 60)

        self.collect_phishing_sms(1500)
        self.collect_phishing_email(1500)
        self.collect_phishing_link(1000)
        self.collect_normal_sms(1500)
        self.collect_normal_email(1000)
        self.collect_normal_general(1500)

        logger.info(f"\nCollection Summary:")
        logger.info(f"  Total phishing: {self.stats['total_phishing']}")
        logger.info(f"    SMS: {self.stats['phishing_sms']}")
        logger.info(f"    Email: {self.stats['phishing_email']}")
        logger.info(f"    Link: {self.stats['phishing_link']}")
        logger.info(f"  Total normal: {self.stats['total_normal']}")
        logger.info(f"    SMS: {self.stats['normal_sms']}")
        logger.info(f"    Email: {self.stats['normal_email']}")
        logger.info(f"    General: {self.stats['normal_general']}")
        logger.info(f"  Grand total: {len(self.collected)}")

        return pd.DataFrame(self.collected)


class DataPreprocessor:

    def __init__(self, existing_df):
        self.existing_texts = set(existing_df['text'].dropna().tolist())
        self.existing_df = existing_df

    def process(self, new_df):
        logger.info("=" * 60)
        logger.info("  Data Preprocessing")
        logger.info("=" * 60)

        initial_count = len(new_df)
        logger.info(f"Initial count: {initial_count}")

        new_df = new_df.dropna(subset=['text'])
        logger.info(f"After dropna(text): {len(new_df)}")

        new_df = new_df[new_df['text'].str.strip().str.len() >= 2]
        logger.info(f"After removing short texts: {len(new_df)}")

        new_df = new_df[new_df['text'].str.strip().str.len() <= 2000]
        logger.info(f"After removing long texts (>2000): {len(new_df)}")

        before_dedup = len(new_df)
        new_df = new_df[~new_df['text'].isin(self.existing_texts)]
        logger.info(f"After cross-dataset dedup: {len(new_df)} (removed {before_dedup - len(new_df)})")

        before_self_dedup = len(new_df)
        new_df = new_df.drop_duplicates(subset=['text'])
        logger.info(f"After self-dedup: {len(new_df)} (removed {before_self_dedup - len(new_df)})")

        new_df['label'] = new_df['label'].astype(int)
        new_df = new_df[new_df['label'].isin([0, 1])]

        new_df['text'] = new_df['text'].str.strip()
        new_df['url'] = new_df['url'].fillna('')
        new_df['scenario'] = new_df['scenario'].fillna('general')

        for col in EXISTING_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = ''

        new_df = new_df[EXISTING_COLUMNS]
        new_df = new_df.reset_index(drop=True)

        logger.info(f"Final count: {len(new_df)}")
        return new_df


class QualityValidator:

    def validate(self, df, existing_df):
        logger.info("=" * 60)
        logger.info("  Quality Validation")
        logger.info("=" * 60)

        results = {}

        results['label_balance'] = df['label'].value_counts().to_dict()
        logger.info(f"Label balance: {results['label_balance']}")

        results['scenario_coverage'] = df['scenario'].value_counts().to_dict()
        logger.info(f"Scenario coverage: {results['scenario_coverage']}")

        new_lens = df['text'].str.len()
        exist_lens = existing_df['text'].str.len()
        results['text_length_new'] = {'mean': new_lens.mean(), 'std': new_lens.std(),
                                       'min': int(new_lens.min()), 'max': int(new_lens.max())}
        results['text_length_existing'] = {'mean': exist_lens.mean(), 'std': exist_lens.std(),
                                            'min': int(exist_lens.min()), 'max': int(exist_lens.max())}
        logger.info(f"New text length: mean={new_lens.mean():.1f}, std={new_lens.std():.1f}")
        logger.info(f"Existing text length: mean={exist_lens.mean():.1f}, std={exist_lens.std():.1f}")

        phishing = df[df['label'] == 1]
        keywords = ['支付宝', '微信', '银行', '账户', '安全', '风险', '解冻', '验证',
                    '密码', '登录', '中奖', '领取', '紧急', '点击', '冻结', '逾期']
        kw_coverage = sum(1 for kw in keywords if phishing['text'].str.contains(kw, na=False).any())
        results['keyword_coverage'] = f"{kw_coverage}/{len(keywords)}"
        logger.info(f"Keyword coverage: {kw_coverage}/{len(keywords)}")

        results['null_check'] = {
            'text_null': int(df['text'].isna().sum()),
            'label_null': int(df['label'].isna().sum()),
        }
        results['duplicate_check'] = int(df['text'].duplicated().sum())
        logger.info(f"Null texts: {results['null_check']['text_null']}")
        logger.info(f"Duplicates: {results['duplicate_check']}")

        overlap = set(df['text']).intersection(set(existing_df['text']))
        results['overlap_with_existing'] = len(overlap)
        logger.info(f"Overlap with existing: {len(overlap)}")

        results['passed'] = (
            results['null_check']['text_null'] == 0 and
            results['null_check']['label_null'] == 0 and
            results['duplicate_check'] == 0 and
            results['overlap_with_existing'] == 0 and
            kw_coverage >= len(keywords) * 0.5
        )
        logger.info(f"Quality validation: {'PASSED' if results['passed'] else 'FAILED'}")
        return results


def main():
    logger.info("=" * 60)
    logger.info("  SUPPLEMENTARY DATA COLLECTION PIPELINE")
    logger.info("=" * 60)

    existing_path = os.path.join(DATA_DIR, 'dataset_20260411_chifraud.csv')
    existing_df = pd.read_csv(existing_path)
    logger.info(f"Existing dataset: {existing_df.shape}")

    collector = SupplementaryDataCollector()
    new_df = collector.collect_all()

    preprocessor = DataPreprocessor(existing_df)
    processed_df = preprocessor.process(new_df)

    validator = QualityValidator()
    validation_results = validator.validate(processed_df, existing_df)

    combined_df = pd.concat([existing_df, processed_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['text'])
    combined_df = combined_df.reset_index(drop=True)

    timestamp = datetime.now().strftime('%Y%m%d')
    combined_path = os.path.join(DATA_DIR, f'dataset_{timestamp}_expanded.csv')
    combined_df.to_csv(combined_path, index=False, encoding='utf-8')
    logger.info(f"\nExpanded dataset saved: {combined_path}")
    logger.info(f"  Original: {len(existing_df)}")
    logger.info(f"  New added: {len(processed_df)}")
    logger.info(f"  Combined: {len(combined_df)}")
    logger.info(f"  Label distribution: {combined_df['label'].value_counts().to_dict()}")
    logger.info(f"  Scenario distribution: {combined_df['scenario'].value_counts().to_dict()}")

    report = {
        'collection_date': datetime.now().isoformat(),
        'existing_samples': len(existing_df),
        'new_samples_collected': len(collector.collected),
        'new_samples_after_preprocessing': len(processed_df),
        'combined_samples': len(combined_df),
        'collection_stats': collector.stats,
        'validation_results': validation_results,
        'combined_label_distribution': combined_df['label'].value_counts().to_dict(),
        'combined_scenario_distribution': combined_df['scenario'].value_counts().to_dict(),
        'output_file': combined_path,
    }

    report_path = os.path.join(OUTPUT_DIR, 'data_collection_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")

    logger.info("\n" + "=" * 60)
    logger.info("  DATA COLLECTION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
