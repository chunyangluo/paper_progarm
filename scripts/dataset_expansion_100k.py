import os
import sys
import json
import random
import hashlib
import logging
import re
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'versions')
RAW_DIR = os.path.join(PROJECT_ROOT, 'src', 'data', 'raw', 'chifraud', 'dataset')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

EXISTING_COLUMNS = ['url', 'phish_id', 'submission_time', 'verification_time',
                    'online', 'target', 'label', 'source', 'text', 'scenario',
                    'timestamp', 'rank']

CHIFRAUD_LABEL_MAP = {
    0: ('normal', 'general'),
    1: ('phishing', 'gambling'),
    2: ('phishing', 'sexual'),
    3: ('phishing', 'fake_id'),
    4: ('phishing', 'fake_card'),
    5: ('phishing', 'drug_dealing'),
    6: ('phishing', 'illegal_cashout'),
    7: ('phishing', 'fake_certificate'),
    8: ('phishing', 'fake_phonecard'),
    9: ('phishing', 'underground_loan'),
    10: ('phishing', 'new_type'),
}

SCENARIO_KEYWORDS = {
    'sms': ['验证码', '短信', '【', '】', '点击', '回复', '退订', '客服', '拨打', '号码'],
    'email': ['尊敬的', '亲爱的', '此致', '邮件', '附件', '抄送', '回复', '发件人', '收件人'],
    'link': ['http', 'https', 'www.', '.com', '.cn', '.net', '链接', '网址', '点击访问'],
    'app': ['APP', '应用', '下载', '安装', '更新', '版本', '扫码', '二维码'],
}

random.seed(42)
np.random.seed(42)


class ChiFraudLoader:

    def load_all(self):
        logger.info("=" * 60)
        logger.info("  Phase 1: Loading ChiFraud Full Dataset")
        logger.info("=" * 60)

        all_dfs = []
        files = {
            'ChiFraud_train.csv': 'chifraud_train',
            'ChiFraud_t2022.csv': 'chifraud_t2022',
            'ChiFraud_t2023.csv': 'chifraud_t2023',
        }

        for fname, source_name in files.items():
            fpath = os.path.join(RAW_DIR, fname)
            if not os.path.exists(fpath):
                logger.warning(f"  {fname} not found, skipping")
                continue
            df = pd.read_csv(fpath, sep='\t')
            df.columns = ['label_id', 'text']
            df['label_id'] = df['label_id'].astype(int)
            df['source'] = source_name

            fraud_count = (df['label_id'] != 0).sum()
            normal_count = (df['label_id'] == 0).sum()
            logger.info(f"  {fname}: {len(df)} rows (fraud={fraud_count}, normal={normal_count})")
            all_dfs.append(df)

        if not all_dfs:
            logger.error("  No ChiFraud files found!")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined['is_phishing'] = (combined['label_id'] != 0).astype(int)

        label_info = combined['label_id'].value_counts().sort_index().to_dict()
        logger.info(f"  Combined ChiFraud: {len(combined)} rows")
        logger.info(f"  Label distribution: {label_info}")

        return combined

    def assign_scenario(self, text):
        text = str(text)
        scores = {}
        for scenario, keywords in SCENARIO_KEYWORDS.items():
            scores[scenario] = sum(1 for kw in keywords if kw in text)
        if max(scores.values()) == 0:
            return 'general'
        return max(scores, key=scores.get)

    def process_for_merge(self, df, max_phishing=50000, max_normal=50000):
        logger.info("  Processing ChiFraud data for merge...")

        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip().str.len() >= 2]
        df = df[df['text'].str.strip().str.len() <= 5000]
        df = df.drop_duplicates(subset=['text'])

        phishing_df = df[df['is_phishing'] == 1].copy()
        normal_df = df[df['is_phishing'] == 0].copy()

        if len(phishing_df) > max_phishing:
            phishing_df = phishing_df.sample(n=max_phishing, random_state=42)
        if len(normal_df) > max_normal:
            normal_df = normal_df.sample(n=max_normal, random_state=42)

        logger.info(f"  Phishing: {len(phishing_df)}, Normal: {len(normal_df)}")

        result_dfs = []
        for subset_df, label_val in [(phishing_df, 1), (normal_df, 0)]:
            records = []
            for _, row in subset_df.iterrows():
                scenario = self.assign_scenario(row['text'])
                sub_type = CHIFRAUD_LABEL_MAP.get(row['label_id'], ('phishing', 'unknown'))[1]
                records.append({
                    'url': '', 'phish_id': '', 'submission_time': '',
                    'verification_time': '', 'online': '', 'target': '',
                    'label': label_val,
                    'source': f"chifraud_{row['source']}_{sub_type}",
                    'text': str(row['text']).strip(),
                    'scenario': scenario,
                    'timestamp': datetime.now().isoformat(),
                    'rank': '',
                })
            result_dfs.append(pd.DataFrame(records))

        result = pd.concat(result_dfs, ignore_index=True)
        logger.info(f"  Processed ChiFraud: {len(result)} rows")
        return result


class TemplateGenerator:

    def __init__(self):
        self.existing_texts = set()

    def set_existing(self, texts):
        self.existing_texts = set(str(t) for t in texts)

    def _gen_if_new(self, text, label, scenario, source, url=''):
        if text in self.existing_texts:
            return None
        self.existing_texts.add(text)
        return {
            'url': url, 'phish_id': '', 'submission_time': '',
            'verification_time': '', 'online': '', 'target': '',
            'label': label, 'source': source, 'text': text,
            'scenario': scenario, 'timestamp': datetime.now().isoformat(), 'rank': '',
        }

    def generate_bank_normal(self, target=5000):
        logger.info(f"  Generating {target} bank/payment normal notifications...")
        templates = [
            "【{bank}】您尾号{tail}的账户于{time}收入{amount}元，余额{balance}元。",
            "【{bank}】您尾号{tail}的账户于{time}支出{amount}元（{merchant}），余额{balance}元。",
            "【{bank}】您尾号{tail}的信用卡于{date}消费{amount}元，可用额度{balance}元。",
            "【{bank}】您尾号{tail}的信用卡本期账单{amount}元，最低还款{min_amount}元，还款日{date}。",
            "【{bank}】您尾号{tail}的账户{date}利息{amount}元已入账。",
            "【{bank}】您预约的{service}已受理，预计{days}个工作日完成，请留意账户变动。",
            "【{bank}】您尾号{tail}的定期存款{amount}元将于{date}到期，请及时处理。",
            "【{bank}】您的{card_type}已成功办理，将于{days}个工作日内寄出，请注意查收。",
            "【{bank}】您尾号{tail}的账户已成功开通{service}功能。",
            "【{bank}】您的贷款申请已审批通过，额度{amount}元，利率{rate}%，请登录APP查看详情。",
            "【{bank}】您尾号{tail}的账户于{time}转账{amount}元至{target_name}，余额{balance}元。",
            "【{bank}】您尾号{tail}的账户已成功修改{service}密码，如非本人操作请拨打{phone}。",
            "【{bank}】您的{service}业务已办理成功，生效日期{date}。",
            "【{bank}】您尾号{tail}的账户{service}已关闭，如有疑问请致电{phone}。",
            "【{bank}】您尾号{tail}的账户月度对账单已生成，请登录网银查看。",
            "【{bank}】您尾号{tail}的账户已绑定{device}设备，如非本人操作请致电{phone}。",
            "【{bank}】您的{service}申请已提交，预计{days}个工作日内审核完成。",
            "【{bank}】您尾号{tail}的账户已设置{service}限额{amount}元/日。",
            "【{bank}】您的{card_type}年费{amount}元已扣除，如需减免请登录APP申请。",
            "【{bank}】您尾号{tail}的账户已成功转入{amount}元（{source_name}），余额{balance}元。",
        ]
        banks = ['工商银行', '建设银行', '农业银行', '中国银行', '招商银行', '交通银行',
                 '邮储银行', '中信银行', '光大银行', '民生银行', '兴业银行', '浦发银行',
                 '平安银行', '华夏银行', '广发银行', '北京银行', '上海银行', '南京银行',
                 '支付宝', '微信支付', '银联', '云闪付']
        services = ['手机银行', '网上银行', '短信提醒', '快捷支付', '自动还款', '定投',
                    '余额提醒', '大额转账', '跨境汇款', '理财购买']
        card_types = ['信用卡', '借记卡', '储蓄卡', '金卡', '白金卡', '钻石卡']
        merchants = ['超市购物', '餐饮消费', '网购支付', '水电缴费', '加油消费',
                     '医院挂号', '学费缴纳', '保险缴费', '物业费', '话费充值']

        samples = []
        attempts = 0
        while len(samples) < target and attempts < target * 10:
            attempts += 1
            template = random.choice(templates)
            text = template.format(
                bank=random.choice(banks),
                tail=f"{random.randint(1000, 9999)}",
                time=f"{random.randint(1,28)}日{random.randint(0,23):02d}:{random.randint(0,59):02d}",
                amount=f"{random.randint(1,50000)}.{random.randint(0,99):02d}",
                balance=f"{random.randint(100,999999)}.{random.randint(0,99):02d}",
                min_amount=f"{random.randint(10,5000)}.{random.randint(0,99):02d}",
                date=f"{random.randint(1,28)}日",
                days=random.choice(['3', '5', '7', '10', '15']),
                service=random.choice(services),
                card_type=random.choice(card_types),
                merchant=random.choice(merchants),
                rate=f"{random.randint(3,8)}.{random.randint(0,99):02d}",
                target_name=random.choice(['张三', '李四', '王五', '赵六', '本人他行账户']),
                source_name=random.choice(['工资收入', '转账汇款', '退款', '理财赎回', '红包']),
                phone=f"955{random.randint(10,99)}",
                device=random.choice(['iPhone', '华为', '小米', 'OPPO', 'vivo']),
            )
            sample = self._gen_if_new(text, 0, 'sms', 'template_bank_normal')
            if sample:
                samples.append(sample)
        logger.info(f"    Generated {len(samples)} bank normal samples")
        return samples

    def generate_ecommerce_normal(self, target=3000):
        logger.info(f"  Generating {target} e-commerce normal notifications...")
        templates = [
            "【{platform}】您的订单{order_id}已付款成功，商家将在{hours}小时内发货。",
            "【{platform}】您的订单{order_id}已发货，快递公司{courier}，运单号{tracking}。",
            "【{platform}】您的包裹已送达{location}，请及时取件。取件码{code}。",
            "【{platform}】您购买的商品{product}已签收，如有问题请在{days}天内申请售后。",
            "【{platform}】您的退款{amount}元已原路退回，预计{days}个工作日到账。",
            "【{platform}】您有{points}积分即将在{date}过期，可兑换{product}等好礼。",
            "【{platform}】{brand}旗舰店{event}，满{threshold}减{discount}，活动时间{date_range}。",
            "【{platform}】您关注的{product}已降价{amount}元，当前价{price}元。",
            "【{platform}】您的会员将于{date}到期，续费享{discount}折优惠。",
            "【{platform}】您的评价已通过审核，获得{points}积分奖励。",
            "【{platform}】您的换货申请已通过，新商品将在{days}天内发出。",
            "【{platform}】您预约的{product}到货通知：商品已补货，限量{count}件。",
            "【{platform}】您的优惠券{coupon_id}即将在{date}过期，面值{amount}元。",
            "【{platform}】您的售后申请已受理，客服将在{hours}小时内联系您。",
            "【{platform}】您参与的拼团已成功，预计{days}天内发货。",
        ]
        platforms = ['淘宝', '京东', '拼多多', '美团', '饿了么', '盒马', '叮咚买菜',
                     '苏宁', '唯品会', '小红书', '得物', '当当', '网易严选']
        products = ['iPhone16', '华为Mate70', '小米15', 'AirPods', '戴森吹风机',
                    '耐克运动鞋', '优衣库羽绒服', '茅台酒', '雅诗兰黛精华', 'iPad']
        brands = ['Apple', '华为', '小米', 'Nike', 'Adidas', 'UNIQLO', 'Dyson']
        couriers = ['顺丰速运', '京东物流', '中通快递', '圆通速递', '韵达快递', '极兔速递']

        samples = []
        attempts = 0
        while len(samples) < target and attempts < target * 10:
            attempts += 1
            template = random.choice(templates)
            text = template.format(
                platform=random.choice(platforms),
                order_id=f"ORD{random.randint(100000000, 999999999)}",
                hours=random.choice(['24', '48', '72']),
                courier=random.choice(couriers),
                tracking=f"{random.choice(['SF','JD','ZT','YT','YD','JT'])}{random.randint(1000000000, 9999999999)}",
                location=random.choice(['菜鸟驿站', '丰巢快递柜', '小区门口', '公司前台']),
                code=f"{random.randint(100000, 999999)}",
                product=random.choice(products),
                days=random.choice(['7', '15', '30']),
                amount=f"{random.randint(1,5000)}.{random.randint(0,99):02d}",
                points=f"{random.randint(100,50000)}",
                date=f"{random.randint(1,28)}日",
                brand=random.choice(brands),
                event=random.choice(['周年庆', '618大促', '双11', '年货节', '超级品牌日']),
                threshold=f"{random.randint(100,500)}",
                discount=f"{random.randint(20,200)}",
                date_range=f"{random.randint(1,28)}日-{random.randint(1,28)}日",
                price=f"{random.randint(99,9999)}",
                coupon_id=f"CPN{random.randint(10000, 99999)}",
                count=random.choice(['50', '100', '200', '500']),
                hours_val=random.choice(['2', '4', '8', '24']),
            )
            sample = self._gen_if_new(text, 0, 'sms', 'template_ecommerce_normal')
            if sample:
                samples.append(sample)
        logger.info(f"    Generated {len(samples)} e-commerce normal samples")
        return samples

    def generate_social_normal(self, target=2000):
        logger.info(f"  Generating {target} social platform normal notifications...")
        templates = [
            "【{platform}】{user}向您发送了好友请求，点击查看详情。",
            "【{platform}】您有{count}条未读消息，来自{user}。",
            "【{platform}】{user}评论了您的{content}：{comment}",
            "【{platform}】{user}赞了您的{content}。",
            "【{platform}】您的{content}已被{count}人浏览，{like_count}人点赞。",
            "【{platform}】您关注的{user}发布了新{content}，快来查看。",
            "【{platform}】您的账号在新设备上登录，如非本人操作请修改密码。",
            "【{platform}】您参与的{activity}将于{date}开始，请准时参加。",
            "【{platform}】您的{content}审核已通过，已公开展示。",
            "【{platform}】系统升级通知：{date}将进行系统维护，届时部分功能暂不可用。",
            "【{platform}】您收到一条来自{user}的{type}邀请。",
            "【{platform}】您的年度报告已生成，今年您共发布了{count}条{content}。",
        ]
        platforms = ['微信', 'QQ', '微博', '抖音', '快手', '小红书', 'B站', '知乎', '豆瓣']
        users = ['小明', '小红', '老王', '张三', '李四', '王五', '同事小刘', '同学小陈']
        contents = ['动态', '文章', '视频', '照片', '评论', '帖子', '作品', '回答']
        comments = ['写得真好！', '太棒了', '学到了', '收藏了', '分享一下', '有同感']
        activities = ['线上直播', '话题讨论', '抽奖活动', '知识竞赛', '读书会']
        types = ['游戏', '活动', '群聊', '关注', '合作']

        samples = []
        attempts = 0
        while len(samples) < target and attempts < target * 10:
            attempts += 1
            template = random.choice(templates)
            text = template.format(
                platform=random.choice(platforms),
                user=random.choice(users),
                count=random.choice(['1', '3', '5', '10', '20']),
                content=random.choice(contents),
                comment=random.choice(comments),
                like_count=random.choice(['10', '50', '100', '500']),
                date=f"{random.randint(1,28)}日",
                activity=random.choice(activities),
                type=random.choice(types),
            )
            sample = self._gen_if_new(text, 0, 'sms', 'template_social_normal')
            if sample:
                samples.append(sample)
        logger.info(f"    Generated {len(samples)} social normal samples")
        return samples

    def generate_gov_normal(self, target=2000):
        logger.info(f"  Generating {target} government service normal notifications...")
        templates = [
            "【{agency}】您的社保缴费已到账，缴费月份{month}，金额{amount}元。",
            "【{agency}】您的公积金账户余额{amount}元，近期无提取记录。",
            "【{agency}】您的{cert_type}已办理完成，请于{date}前到{location}领取。",
            "【{agency}】您{year}年度个人所得税汇算清缴已完成，{result}。",
            "【{agency}】您的居住证将于{date}到期，请及时办理续签。",
            "【{agency}】您的{service}申请已受理，受理编号{case_id}。",
            "【{agency}】您预约的{service}时间为{date} {time}，地点{location}，请携带身份证。",
            "【{agency}】您的车辆{plate}已通过年检，有效期至{date}。",
            "【{agency}】您的水费账单{amount}元，缴费截止{date}。",
            "【{agency}】您的{service}办理进度：已进入审核阶段，预计{days}个工作日完成。",
            "【{agency}】您的医保账户已成功绑定{hospital}，可直接结算。",
            "【{agency}】您的{cert_type}已通过审批，电子证照已生成，请登录APP查看。",
        ]
        agencies = ['社保局', '公积金中心', '税务局', '公安局', '民政局', '卫健委',
                    '交通局', '人社局', '不动产中心', '医保局', '城管局', '教育局']
        cert_types = ['身份证', '户口本', '结婚证', '营业执照', '驾驶证', '居住证', '出生证明']
        services = ['社保查询', '公积金提取', '个税申报', '居住证办理', '医保报销', '不动产登记']
        locations = ['市民服务中心', '政务大厅', '街道办事处', '社区服务中心']

        samples = []
        attempts = 0
        while len(samples) < target and attempts < target * 10:
            attempts += 1
            template = random.choice(templates)
            text = template.format(
                agency=random.choice(agencies),
                month=f"{random.randint(1,12)}月",
                amount=f"{random.randint(100,10000)}.{random.randint(0,99):02d}",
                date=f"{random.randint(1,28)}日",
                cert_type=random.choice(cert_types),
                location=random.choice(locations),
                year=f"20{random.randint(20,25)}",
                result=random.choice(['应退税额0元', '应补税额0元', '已退税{amount}元'.format(amount=random.randint(100,5000))]),
                service=random.choice(services),
                case_id=f"CASE{random.randint(100000, 999999)}",
                time=f"{random.randint(8,17)}:{random.randint(0,59):02d}",
                plate=f"京{chr(random.randint(65,90))}{random.randint(10000,99999)}",
                days=random.choice(['5', '10', '15', '20']),
                hospital=random.choice(['协和医院', '同仁医院', '北医三院', '朝阳医院']),
            )
            sample = self._gen_if_new(text, 0, 'sms', 'template_gov_normal')
            if sample:
                samples.append(sample)
        logger.info(f"    Generated {len(samples)} government normal samples")
        return samples

    def generate_phishing_variants(self, target=8000):
        logger.info(f"  Generating {target} phishing variant samples...")
        templates = [
            "【{platform}】紧急通知：您的{account}在{location}被异常登录，请立即验证身份 {url}",
            "【{platform}】您的{account}已被{action}，请点击 {url} 进行{operation}，否则{consequence}",
            "【{platform}】系统检测到您的{account}存在{risk_type}，请立即处理 {url}",
            "【{platform}】尊敬的用户，您的{account}即将{status}，请点击 {url} 确认保留",
            "【{platform}】恭喜！您被选中为幸运用户，获得{reward}，请点击 {url} 领取",
            "【{platform}】您的{account}有{amount}元待领取，请点击 {url} 确认到账",
            "【{platform}】重要：您的{account}密码已泄露，请立即修改 {url}",
            "【{platform}】您的{account}涉嫌{violation}，请点击 {url} 配合调查",
            "【{platform}】您的{service}即将到期，续费享{discount}折优惠，点击 {url}",
            "【{platform}】您的{account}被临时{status}，请点击 {url} 申请恢复",
            "【{platform}】您有一笔{amount}元的退款待确认，请点击 {url} 办理",
            "【{platform}】您的{account}积分{points}即将清零，点击 {url} 立即兑换",
            "【{platform}】您的{account}实名认证已过期，请点击 {url} 重新认证",
            "【{platform}】紧急！您的{account}在{time}发生{transaction}，如非本人操作请点击 {url}",
            "【{platform}】您的{account}已被列入{list_type}，请点击 {url} 申诉",
            "【{platform}】您的{service}资格即将被取消，点击 {url} 保住资格",
            "【{platform}】您的{account}存在{risk_type}，请在{deadline}前点击 {url} 处理",
            "【{platform}】您的{account}已被{action}，需要{fee}元解封费，点击 {url} 缴费",
            "【{platform}】您的{account}收到{amount}元转账，需{fee}元手续费方可到账，点击 {url}",
            "【{platform}】尊敬的VIP用户，您有{reward}待领取，仅限今日，点击 {url}",
        ]
        platforms = ['支付宝', '微信', '银行', '工商银行', '建设银行', '农业银行', '中国银行',
                     '招商银行', '京东', '淘宝', '美团', '抖音', '12306', '中国移动',
                     '顺丰快递', '社保局', '税务局', '公积金中心', '银联', '花呗',
                     '借呗', '京东白条', '度小满', '微粒贷', '360借条']
        accounts = ['账户', '账号', '支付账户', '银行卡', '信用卡', '会员', '积分账户',
                    '社保账户', '公积金账户', '手机号', '身份信息', '信用额度', '贷款账户']
        actions = ['冻结', '限制', '锁定', '暂停', '封禁']
        operations = ['解冻', '解封', '恢复', '验证', '确认']
        consequences = ['永久冻结', '自动注销', '影响征信', '无法恢复', '降低额度']
        risk_types = ['安全风险', '异常登录', '被盗风险', '信息泄露', '违规操作', '可疑交易']
        statuses = ['过期', '被注销', '被降级', '被清零', '失效']
        rewards = ['iPhone16大奖', '现金红包888元', '免费旅游券', '购物免单资格',
                   'VIP会员', '100元话费', '50元优惠券', '限量版礼品']
        violations = ['洗钱', '违规交易', '套现', '欺诈', '信息不实']
        services = ['会员', 'VIP', '保险', '理财', '贷款', '信用卡']
        list_types = ['黑名单', '风险名单', '异常名单', '监控名单']
        locations = ['异地', '境外', '未知设备', '非常用IP', '新设备']
        transactions = ['异常转账', '大额消费', '可疑提现', '未知支付']
        domains = ['verify', 'secure', 'auth', 'login', 'confirm', 'update', 'check',
                   'safe', 'protect', 'reset', 'unlock', 'service']
        suffixes = ['.cn', '.com', '.net', '.xyz', '.top', '.cc', '.vip', '.tk']

        samples = []
        attempts = 0
        while len(samples) < target and attempts < target * 15:
            attempts += 1
            template = random.choice(templates)
            domain = random.choice(domains)
            suffix = random.choice(suffixes)
            url = f"https://{random.choice(platforms).lower()}-{domain}{suffix}/{hashlib.md5(str(random.random()).encode()).hexdigest()[:6]}"
            text = template.format(
                platform=random.choice(platforms),
                account=random.choice(accounts),
                action=random.choice(actions),
                operation=random.choice(operations),
                consequence=random.choice(consequences),
                risk_type=random.choice(risk_types),
                status=random.choice(statuses),
                reward=random.choice(rewards),
                amount=f"{random.randint(100,50000)}.{random.randint(0,99):02d}",
                url=url,
                violation=random.choice(violations),
                service=random.choice(services),
                discount=random.choice(['5', '3', '2', '1']),
                points=f"{random.randint(1000,50000)}",
                deadline=random.choice(['24小时', '48小时', '3天', '7天']),
                fee=f"{random.randint(50,2000)}.{random.randint(0,99):02d}",
                location=random.choice(locations),
                time=f"{random.randint(1,28)}日{random.randint(0,23):02d}:{random.randint(0,59):02d}",
                transaction=random.choice(transactions),
                list_type=random.choice(list_types),
            )
            scenario = 'sms' if '【' in text else 'email'
            sample = self._gen_if_new(text, 1, scenario, 'template_phishing_variant', url=url)
            if sample:
                samples.append(sample)
        logger.info(f"    Generated {len(samples)} phishing variant samples")
        return samples

    def generate_phishing_email_extended(self, target=3000):
        logger.info(f"  Generating {target} phishing email samples...")
        templates = [
            "尊敬的{title}，\n\n您的{account}于{time}在{location}进行了{operation}。为保障安全，请立即登录 {url} 确认身份。\n\n如非本人操作，请立即修改密码。\n\n{platform}安全中心",
            "亲爱的{title}，\n\n恭喜您被选为{platform}{event}幸运用户！您将获得{reward}，请于{deadline}前点击 {url} 领取。\n\n逾期视为自动放弃。\n\n{platform}运营中心",
            "尊敬的{title}，\n\n根据{authority}要求，您的{account}需要进行{operation}。请于{deadline}前登录 {url} 完成{operation}。\n\n逾期未处理将{consequence}。\n\n{authority}通知中心",
            "尊敬的{title}，\n\n您的{account}已被临时{status}，原因是{reason}。如需恢复，请点击 {url} 提交申诉。\n\n{platform}安全团队",
            "亲爱的{title}，\n\n您的{account}有一笔{amount}元的{transaction_type}待确认。请点击 {url} 确认到账。\n\n需支付{fee}元手续费。\n\n{platform}财务中心",
            "尊敬的{title}，\n\n您的{account}信用评分已降至{score}分，将影响您的{impact}。请点击 {url} 立即修复信用。\n\n{platform}风控中心",
        ]
        titles = ['用户', '客户', '会员', '先生/女士']
        accounts = ['银行账户', '信用卡', '支付宝', '微信支付', '网银', '投资账户', '贷款账户']
        platforms = ['中国银行', '工商银行', '建设银行', '支付宝', '微信支付', '京东金融', '银联']
        operations = ['身份验证', '安全确认', '信息更新', '密码重置', '账户解冻']
        authorities = ['银保监会', '人民银行', '网信办', '公安部', '税务局']
        statuses = ['冻结', '限制', '锁定', '暂停']
        reasons = ['异常登录', '可疑交易', '信息变更', '安全审查']
        transaction_types = ['退款', '转账', '奖金', '补贴', '理赔款']
        impacts = ['贷款审批', '信用卡申请', '出行购票', '高消费']

        samples = []
        attempts = 0
        while len(samples) < target and attempts < target * 10:
            attempts += 1
            template = random.choice(templates)
            domain = random.choice(['verify', 'secure', 'auth', 'service', 'help'])
            suffix = random.choice(['.cn', '.com', '.net', '.xyz'])
            url = f"https://{domain}-bank{suffix}/{hashlib.md5(str(random.random()).encode()).hexdigest()[:8]}"
            text = template.format(
                title=random.choice(titles),
                account=random.choice(accounts),
                time=f"{random.randint(1,28)}日{random.randint(0,23):02d}:{random.randint(0,59):02d}",
                location=random.choice(['异地', '境外', '未知设备', '新设备']),
                operation=random.choice(operations),
                url=url,
                platform=random.choice(platforms),
                event=random.choice(['年度', '季度', '周年庆', '限时']),
                reward=random.choice(['现金奖励888元', '积分奖励5000分', '免息额度10000元', 'VIP权益']),
                deadline=random.choice(['24小时', '48小时', '3天', '7天']),
                authority=random.choice(authorities),
                consequence=random.choice(['永久冻结', '自动注销', '限制使用', '影响征信']),
                status=random.choice(statuses),
                reason=random.choice(reasons),
                amount=f"{random.randint(1000,50000)}.{random.randint(0,99):02d}",
                transaction_type=random.choice(transaction_types),
                fee=f"{random.randint(50,500)}.{random.randint(0,99):02d}",
                score=f"{random.randint(300,550)}",
                impact=random.choice(impacts),
            )
            sample = self._gen_if_new(text, 1, 'email', 'template_phishing_email', url=url)
            if sample:
                samples.append(sample)
        logger.info(f"    Generated {len(samples)} phishing email samples")
        return samples

    def generate_hard_boundary_samples(self, target=3000):
        logger.info(f"  Generating {target} hard/boundary samples...")
        phishing_templates = [
            "【{bank}】您尾号{tail}的账户于{time}发生{amount}元{transaction}，如非本人操作请拨打{phone}或点击 {url} 冻结账户。",
            "【{bank}】检测到您的账户存在安全风险，请立即登录 {url} 进行身份验证，否则将临时冻结账户。",
            "【{bank}】您的信用卡已成功消费{amount}元（{merchant}），当前可用额度{balance}元。如非本人操作请点击 {url}。",
            "【{platform}】您的账户在新设备上登录，为保障安全请点击 {url} 确认身份。",
            "【{bank}】您尾号{tail}的账户需更新安全信息，请点击 {url} 完成验证，逾期将限制交易。",
        ]
        normal_templates = [
            "【{bank}】您尾号{tail}的账户于{time}发生{amount}元{transaction}，余额{balance}元。如有疑问请拨打{phone}。",
            "【{bank}】为保障您的账户安全，请定期修改登录密码。如需帮助请拨打{phone}。",
            "【{bank}】您尾号{tail}的信用卡消费{amount}元（{merchant}），可用额度{balance}元。",
            "【{platform}】您的账号在新设备上登录成功，如非本人操作请修改密码。",
            "【{bank}】您的账户安全等级为{level}，建议开启{service}功能提升安全等级，请登录APP操作。",
        ]
        banks = ['工商银行', '建设银行', '农业银行', '中国银行', '招商银行', '支付宝', '微信支付']
        transactions = ['消费', '转账', '取款', '支付']
        merchants = ['超市购物', '餐饮消费', '网购支付', '医院缴费']
        platforms = ['微信', '支付宝', '京东', '淘宝']

        samples = []
        attempts = 0
        half = target // 2
        while len(samples) < target and attempts < target * 15:
            attempts += 1
            is_phishing = len(samples) < half
            templates = phishing_templates if is_phishing else normal_templates
            template = random.choice(templates)
            domain = random.choice(['verify', 'secure', 'auth'])
            suffix = random.choice(['.cn', '.com'])
            url = f"https://{domain}-bank{suffix}/{hashlib.md5(str(random.random()).encode()).hexdigest()[:6]}"
            text = template.format(
                bank=random.choice(banks),
                tail=f"{random.randint(1000, 9999)}",
                time=f"{random.randint(1,28)}日{random.randint(0,23):02d}:{random.randint(0,59):02d}",
                amount=f"{random.randint(10,50000)}.{random.randint(0,99):02d}",
                transaction=random.choice(transactions),
                balance=f"{random.randint(100,999999)}.{random.randint(0,99):02d}",
                phone=f"955{random.randint(10,99)}",
                url=url,
                merchant=random.choice(merchants),
                platform=random.choice(platforms),
                level=random.choice(['低', '中', '较高']),
                service=random.choice(['双重验证', '手势密码', '指纹登录', '人脸识别']),
            )
            label = 1 if is_phishing else 0
            source = 'template_hard_phishing' if is_phishing else 'template_hard_normal'
            sample = self._gen_if_new(text, label, 'sms', source, url=url if is_phishing else '')
            if sample:
                samples.append(sample)
        logger.info(f"    Generated {len(samples)} hard boundary samples")
        return samples

    def generate_all(self):
        logger.info("=" * 60)
        logger.info("  Phase 2: Template-Based Generation")
        logger.info("=" * 60)

        all_samples = []
        all_samples.extend(self.generate_bank_normal(5000))
        all_samples.extend(self.generate_ecommerce_normal(3000))
        all_samples.extend(self.generate_social_normal(2000))
        all_samples.extend(self.generate_gov_normal(2000))
        all_samples.extend(self.generate_phishing_variants(8000))
        all_samples.extend(self.generate_phishing_email_extended(3000))
        all_samples.extend(self.generate_hard_boundary_samples(3000))

        logger.info(f"  Total template samples: {len(all_samples)}")
        return pd.DataFrame(all_samples)


class DataAugmenter:

    SYNONYM_DICT = {
        '账户': ['账号', '用户账户', '账户信息'], '账号': ['账户', '用户名', '登录名'],
        '安全': ['安全保障', '账户安全', '安全防护'], '风险': ['安全隐患', '安全风险', '潜在风险'],
        '验证': ['身份验证', '安全验证', '身份确认'], '密码': ['登录密码', '安全密码', '支付密码'],
        '登录': ['登录系统', '账户登录', '在线登录'], '冻结': ['临时冻结', '安全冻结', '账户冻结'],
        '解冻': ['解除冻结', '恢复使用', '账户解冻'], '领取': ['立即领取', '尽快领取', '点击领取'],
        '紧急': ['紧急通知', '紧急提醒', '重要通知'], '立即': ['马上', '尽快', '立刻'],
        '点击': ['点击链接', '访问', '打开'], '确认': ['确认身份', '核实', '验证确认'],
        '过期': ['即将过期', '即将失效', '到期'], '异常': ['异常情况', '异常状态', '不正常'],
        '限制': ['临时限制', '使用限制', '功能限制'], '更新': ['信息更新', '资料更新', '及时更新'],
    }

    def synonym_replace(self, text, n=1):
        words = list(self.SYNONYM_DICT.keys())
        random.shuffle(words)
        replaced = 0
        for word in words:
            if word in text and replaced < n:
                syn = random.choice(self.SYNONYM_DICT[word])
                text = text.replace(word, syn, 1)
                replaced += 1
        return text

    def random_swap_sentences(self, text):
        sentences = re.split(r'([。！？\n])', text)
        if len(sentences) <= 2:
            return text
        parts = []
        for i in range(0, len(sentences) - 1, 2):
            parts.append(sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else ''))
        if len(parts) <= 1:
            return text
        idx1, idx2 = random.sample(range(len(parts)), 2)
        parts[idx1], parts[idx2] = parts[idx2], parts[idx1]
        return ''.join(parts)

    def random_insert(self, text, n=1):
        insert_words = ['请', '务必', '及时', '尽快', '注意', '重要', '务必', '一定']
        for _ in range(n):
            pos = random.randint(0, max(0, len(text) - 1))
            word = random.choice(insert_words)
            text = text[:pos] + word + text[pos:]
        return text

    def random_delete_char(self, text, n=1):
        for _ in range(n):
            if len(text) > 10:
                pos = random.randint(0, len(text) - 1)
                text = text[:pos] + text[pos + 1:]
        return text

    def augment_sample(self, row):
        text = str(row['text'])
        label = row['label']
        scenario = row['scenario']
        source = row['source']
        url = row.get('url', '')

        augmented = []
        methods = [
            ('synonym', lambda t: self.synonym_replace(t, random.randint(1, 2))),
            ('swap', lambda t: self.random_swap_sentences(t)),
            ('insert', lambda t: self.random_insert(t, 1)),
            ('delete', lambda t: self.random_delete_char(t, 1)),
        ]

        for method_name, method_func in methods:
            try:
                new_text = method_func(text)
                if new_text != text and len(new_text) >= 2:
                    augmented.append({
                        'url': url, 'phish_id': '', 'submission_time': '',
                        'verification_time': '', 'online': '', 'target': '',
                        'label': label, 'source': f'aug_{method_name}_{source}',
                        'text': new_text, 'scenario': scenario,
                        'timestamp': datetime.now().isoformat(), 'rank': '',
                    })
            except:
                pass

        return augmented

    def augment_batch(self, df, target_count, existing_texts):
        logger.info(f"  Augmenting {len(df)} samples, target: {target_count}...")

        all_augmented = []
        existing_set = set(str(t) for t in existing_texts)
        indices = list(range(len(df)))
        random.shuffle(indices)

        idx = 0
        while len(all_augmented) < target_count and idx < len(indices) * 3:
            i = indices[idx % len(indices)]
            row = df.iloc[i]
            aug_samples = self.augment_sample(row)
            for s in aug_samples:
                if s['text'] not in existing_set and len(all_augmented) < target_count:
                    existing_set.add(s['text'])
                    all_augmented.append(s)
            idx += 1

        logger.info(f"    Generated {len(all_augmented)} augmented samples")
        return pd.DataFrame(all_augmented)


class DataPipeline:

    def __init__(self):
        self.chifraud_loader = ChiFraudLoader()
        self.template_gen = TemplateGenerator()
        self.augmenter = DataAugmenter()
        self.stats = {}

    def run(self):
        logger.info("=" * 60)
        logger.info("  SYSTEMATIC DATASET EXPANSION PIPELINE")
        logger.info("  Target: 100,000+ samples")
        logger.info("=" * 60)

        logger.info("\n[Step 1/7] Loading existing expanded dataset...")
        existing_path = os.path.join(DATA_DIR, 'dataset_20260421_expanded.csv')
        existing_df = pd.read_csv(existing_path)
        logger.info(f"  Existing dataset: {len(existing_df)} rows")
        self.stats['existing_total'] = len(existing_df)
        self.stats['existing_phishing'] = int((existing_df['label'] == 1).sum())
        self.stats['existing_normal'] = int((existing_df['label'] == 0).sum())

        logger.info("\n[Step 2/7] Loading ChiFraud full dataset...")
        chifraud_raw = self.chifraud_loader.load_all()
        chifraud_processed = self.chifraud_loader.process_for_merge(
            chifraud_raw, max_phishing=50000, max_normal=50000
        )
        self.stats['chifraud_phishing'] = int((chifraud_processed['label'] == 1).sum())
        self.stats['chifraud_normal'] = int((chifraud_processed['label'] == 0).sum())

        logger.info("\n[Step 3/7] Generating template-based samples...")
        self.template_gen.set_existing(
            list(existing_df['text']) + list(chifraud_processed['text'])
        )
        template_df = self.template_gen.generate_all()
        self.stats['template_total'] = len(template_df)
        self.stats['template_phishing'] = int((template_df['label'] == 1).sum())
        self.stats['template_normal'] = int((template_df['label'] == 0).sum())

        logger.info("\n[Step 4/7] Merging all data sources...")
        all_dfs = [existing_df, chifraud_processed, template_df]
        for df in all_dfs:
            for col in EXISTING_COLUMNS:
                if col not in df.columns:
                    df[col] = ''
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined[EXISTING_COLUMNS]
        logger.info(f"  Before dedup: {len(combined)}")

        combined['text'] = combined['text'].astype(str).str.strip()
        combined = combined[combined['text'].str.len() >= 2]
        combined = combined[combined['text'].str.len() <= 5000]
        combined = combined.dropna(subset=['text'])
        combined = combined.drop_duplicates(subset=['text'])
        combined = combined.reset_index(drop=True)
        logger.info(f"  After dedup: {len(combined)}")

        phishing_count = (combined['label'] == 1).sum()
        normal_count = (combined['label'] == 0).sum()
        logger.info(f"  Phishing: {phishing_count}, Normal: {normal_count}")
        self.stats['after_merge_total'] = len(combined)
        self.stats['after_merge_phishing'] = phishing_count
        self.stats['after_merge_normal'] = normal_count

        logger.info("\n[Step 5/7] Data augmentation...")
        phishing_df = combined[combined['label'] == 1]
        normal_df = combined[combined['label'] == 0]

        need_phishing = max(0, 50000 - phishing_count)
        need_normal = max(0, 50000 - normal_count)

        aug_dfs = []
        existing_texts = list(combined['text'])

        if need_phishing > 0:
            logger.info(f"  Need {need_phishing} more phishing samples via augmentation")
            aug_phishing = self.augmenter.augment_batch(phishing_df, need_phishing, existing_texts)
            aug_dfs.append(aug_phishing)
            existing_texts.extend(list(aug_phishing['text']))

        if need_normal > 0:
            logger.info(f"  Need {need_normal} more normal samples via augmentation")
            aug_normal = self.augmenter.augment_batch(normal_df, need_normal, existing_texts)
            aug_dfs.append(aug_normal)
            existing_texts.extend(list(aug_normal['text']))

        if aug_dfs:
            aug_combined = pd.concat(aug_dfs, ignore_index=True)
            combined = pd.concat([combined, aug_combined], ignore_index=True)
            combined = combined.drop_duplicates(subset=['text'])
            combined = combined.reset_index(drop=True)
            self.stats['augmented_phishing'] = need_phishing
            self.stats['augmented_normal'] = need_normal
        else:
            self.stats['augmented_phishing'] = 0
            self.stats['augmented_normal'] = 0

        logger.info(f"  After augmentation: {len(combined)}")
        phishing_count = (combined['label'] == 1).sum()
        normal_count = (combined['label'] == 0).sum()
        logger.info(f"  Phishing: {phishing_count}, Normal: {normal_count}")

        logger.info("\n[Step 6/7] Balancing dataset to 50:50...")
        min_count = min(phishing_count, normal_count)
        if phishing_count > min_count:
            phishing_balanced = combined[combined['label'] == 1].sample(n=min_count, random_state=42)
        else:
            phishing_balanced = combined[combined['label'] == 1]
        if normal_count > min_count:
            normal_balanced = combined[combined['label'] == 0].sample(n=min_count, random_state=42)
        else:
            normal_balanced = combined[combined['label'] == 0]

        balanced = pd.concat([phishing_balanced, normal_balanced], ignore_index=True)
        balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        final_phishing = (balanced['label'] == 1).sum()
        final_normal = (balanced['label'] == 0).sum()
        logger.info(f"  Final: {len(balanced)} (phishing={final_phishing}, normal={final_normal})")
        self.stats['final_total'] = len(balanced)
        self.stats['final_phishing'] = final_phishing
        self.stats['final_normal'] = final_normal

        logger.info("\n[Step 7/7] Saving dataset...")
        timestamp = datetime.now().strftime('%Y%m%d')
        output_path = os.path.join(DATA_DIR, f'dataset_{timestamp}_100k.csv')
        balanced.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"  Saved to: {output_path}")

        self._quality_check(balanced)
        self._save_report(balanced, output_path)

        logger.info("\n" + "=" * 60)
        logger.info(f"  EXPANSION COMPLETE: {len(balanced)} samples")
        logger.info("=" * 60)

        return balanced

    def _quality_check(self, df):
        logger.info("\n  Quality Check:")
        null_text = df['text'].isna().sum()
        null_label = df['label'].isna().sum()
        duplicates = df['text'].duplicated().sum()
        invalid_labels = (~df['label'].isin([0, 1])).sum()
        short_texts = (df['text'].str.len() < 2).sum()

        phishing_kw = ['支付宝', '微信', '银行', '账户', '安全', '风险', '验证',
                       '密码', '登录', '中奖', '冻结', '逾期', '紧急', '点击']
        phishing_df = df[df['label'] == 1]
        kw_coverage = sum(1 for kw in phishing_kw
                         if phishing_df['text'].str.contains(kw, na=False).any())

        lens = df['text'].str.len()
        scenario_dist = df['scenario'].value_counts().to_dict()
        source_dist = df['source'].value_counts().head(20).to_dict()

        checks = {
            'null_text': int(null_text),
            'null_label': int(null_label),
            'duplicates': int(duplicates),
            'invalid_labels': int(invalid_labels),
            'short_texts': int(short_texts),
            'keyword_coverage': f"{kw_coverage}/{len(phishing_kw)}",
            'text_length': {'mean': round(lens.mean(), 1), 'std': round(lens.std(), 1),
                           'min': int(lens.min()), 'max': int(lens.max()),
                           'median': round(lens.median(), 1)},
            'scenario_distribution': scenario_dist,
            'source_distribution_top20': source_dist,
        }

        for k, v in checks.items():
            logger.info(f"    {k}: {v}")

        passed = (null_text == 0 and null_label == 0 and duplicates == 0
                  and invalid_labels == 0 and kw_coverage >= len(phishing_kw) * 0.5)
        logger.info(f"    Overall: {'PASSED' if passed else 'FAILED'}")
        self.stats['quality_check'] = checks
        self.stats['quality_passed'] = passed

    def _save_report(self, df, output_path):
        report = {
            'expansion_date': datetime.now().isoformat(),
            'target': '100,000+',
            'achieved': len(df),
            'stats': self.stats,
            'output_file': output_path,
        }
        report_path = os.path.join(OUTPUT_DIR, 'dataset_expansion_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"  Report saved to: {report_path}")


if __name__ == '__main__':
    pipeline = DataPipeline()
    pipeline.run()
