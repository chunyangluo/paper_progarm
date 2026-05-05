import pandas as pd
import random
import hashlib
from datetime import datetime

EXISTING_COLUMNS = ['url', 'phish_id', 'submission_time', 'verification_time',
                    'online', 'target', 'label', 'source', 'text', 'scenario',
                    'timestamp', 'rank']

df = pd.read_csv(r"c:\Users\chuny\Desktop\paper_progarm\data\versions\dataset_20260421_expanded.csv")
print(f"Current dataset: {len(df)}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")
print(f"Scenario distribution: {df['scenario'].value_counts().to_dict()}")

phishing_count = (df['label'] == 1).sum()
normal_count = (df['label'] == 0).sum()
need_normal = phishing_count - normal_count
print(f"\nNeed {need_normal} more normal samples to balance")

normal_general_templates = [
    "您好，请问这个产品还有库存吗？我想订购一批。",
    "今天的天气真不错，适合出去散步。",
    "请问你们公司的营业时间是怎样的？",
    "我需要一份最新的产品目录，可以发给我吗？",
    "上次讨论的方案我已经修改好了，请查收。",
    "请问这个功能怎么使用？能详细说明一下吗？",
    "我下周要出差，需要提前安排好工作交接。",
    "这个季度的业绩报告已经完成了，请各位审阅。",
    "请问附近有没有好的健身房推荐？",
    "我想了解一下你们的培训课程安排。",
    "最近在读一本关于项目管理的书，收获很大。",
    "请问这个地址怎么走？我第一次来不太熟悉。",
    "明天的会议改到10点了，请注意时间调整。",
    "我已将文件上传到共享文件夹，请各位下载查看。",
    "请问你们提供免费试用吗？我想先体验一下。",
    "这个月的电费比上个月高了不少，需要检查一下。",
    "请问有没有推荐的日语学习资料？",
    "我打算周末去爬山，有什么好的路线推荐吗？",
    "请问这个软件支持Mac系统吗？",
    "我们的项目进展顺利，预计下月可以交付。",
    "请问你们有在线客服吗？我想咨询一些问题。",
    "这个设计稿我看了，整体不错，细节还需要调整。",
    "请问附近哪里可以打印文件？",
    "我已收到货物，质量很好，谢谢！",
    "请问你们有微信群吗？我想加入交流。",
    "这个功能很实用，感谢开发团队的努力。",
    "请问如何修改个人资料？我找不到入口。",
    "今天的会议内容很重要，请大家做好笔记。",
    "请问你们支持货到付款吗？",
    "我已完成了在线学习课程，证书什么时候发放？",
    "请问这个优惠活动什么时候截止？",
    "我们部门的新同事明天入职，请大家多多关照。",
    "请问你们有实体店吗？我想线下体验。",
    "这个季度的KPI完成情况良好，继续保持。",
    "请问如何申请退款？流程是怎样的？",
    "我已提交了报销单，请问多久可以到账？",
    "请问你们有学生优惠吗？",
    "这个项目需要跨部门协作，请各位配合。",
    "请问如何查看我的订单状态？",
    "我已预约了下周的体检，需要注意什么？",
    "请问你们提供安装服务吗？费用多少？",
    "这个月的团建活动定在周六，请大家安排好时间。",
    "请问如何联系售后？我有一个产品问题需要解决。",
    "我已完成了年度总结，请领导审阅。",
    "请问你们有APP吗？在哪里下载？",
    "这个方案的成本预算需要重新核算。",
    "请问如何开通会员？有什么权益？",
    "我已将会议纪要发送到各位邮箱，请查收。",
    "请问你们提供上门服务吗？",
    "这个版本的更新内容我看了，改进很大。",
]

new_samples = []
existing_texts = set(df['text'].tolist())

count = 0
attempts = 0
while count < need_normal and attempts < need_normal * 3:
    template = random.choice(normal_general_templates)
    variations = [
        template,
        template + " 谢谢！",
        template.replace("。", "，麻烦了。"),
        template + " 期待回复。",
    ]
    text = random.choice(variations)
    attempts += 1

    if text in existing_texts:
        continue

    sample = {
        'url': '', 'phish_id': '', 'submission_time': '', 'verification_time': '',
        'online': '', 'target': '', 'label': 0, 'source': 'synthetic_normal_general',
        'text': text, 'scenario': 'general', 'timestamp': datetime.now().isoformat(), 'rank': '',
    }
    new_samples.append(sample)
    existing_texts.add(text)
    count += 1

print(f"\nGenerated {count} additional normal general samples")

new_df = pd.DataFrame(new_samples)
combined = pd.concat([df, new_df], ignore_index=True)
combined = combined.drop_duplicates(subset=['text']).reset_index(drop=True)

output_path = r"c:\Users\chuny\Desktop\paper_progarm\data\versions\dataset_20260421_expanded.csv"
combined.to_csv(output_path, index=False, encoding='utf-8')

print(f"\nFinal dataset: {len(combined)}")
print(f"Label distribution: {combined['label'].value_counts().to_dict()}")
print(f"Scenario distribution: {combined['scenario'].value_counts().to_dict()}")
print(f"Saved to: {output_path}")
