import pandas as pd
import random
from datetime import datetime

EXISTING_COLUMNS = ['url', 'phish_id', 'submission_time', 'verification_time',
                    'online', 'target', 'label', 'source', 'text', 'scenario',
                    'timestamp', 'rank']

df = pd.read_csv(r"c:\Users\chuny\Desktop\paper_progarm\data\versions\dataset_20260421_expanded.csv")
existing_texts = set(df['text'].tolist())
phishing_count = (df['label'] == 1).sum()
normal_count = (df['label'] == 0).sum()
need = phishing_count - normal_count
print(f"Current: phishing={phishing_count}, normal={normal_count}, need={need}")

subjects = ['项目', '工作', '学习', '生活', '健康', '运动', '旅行', '美食', '读书',
           '音乐', '电影', '科技', '设计', '摄影', '编程', '写作', '绘画', '园艺']
verbs = ['讨论', '安排', '计划', '准备', '了解', '分享', '推荐', '学习', '练习',
        '参加', '组织', '完成', '开始', '继续', '尝试', '体验', '探索']
objects = ['方案', '计划', '进度', '内容', '资料', '方法', '技巧', '活动', '课程',
          '任务', '项目', '报告', '总结', '安排', '日程', '目标', '想法']
times = ['今天', '明天', '下周', '这周末', '下个月', '本周', '最近', '近期']
locations = ['会议室', '办公室', '线上', '公司', '学校', '家里', '图书馆', '咖啡厅']
people = ['大家', '各位', '同事们', '同学们', '朋友们', '团队成员']

new_samples = []
count = 0
attempts = 0

while count < need and attempts < need * 5:
    attempts += 1
    patterns = [
        f"请问{random.choice(times)}{random.choice(locations)}的{random.choice(subjects)}{random.choice(verbs)}还有名额吗？",
        f"{random.choice(times)}需要{random.choice(verbs)}一下{random.choice(subjects)}的{random.choice(objects)}，请{random.choice(people)}注意。",
        f"关于{random.choice(subjects)}的{random.choice(objects)}，我已经{random.choice(verbs)}好了，请查收。",
        f"我想{random.choice(verbs)}一下{random.choice(subjects)}相关的内容，有什么{random.choice(objects)}推荐吗？",
        f"{random.choice(people)}好，{random.choice(times)}的{random.choice(subjects)}{random.choice(verbs)}安排在{random.choice(locations)}。",
        f"请问{random.choice(subjects)}的{random.choice(objects)}什么时候可以{random.choice(verbs)}？",
        f"我已完成了{random.choice(subjects)}的{random.choice(objects)}，{random.choice(times)}提交。",
        f"{random.choice(times)}的{random.choice(subjects)}{random.choice(verbs)}已经结束了，感谢{random.choice(people)}的参与。",
        f"请问如何{random.choice(verbs)}{random.choice(subjects)}的{random.choice(objects)}？需要什么材料？",
        f"关于{random.choice(subjects)}，我有一些{random.choice(objects)}想和大家{random.choice(verbs)}。",
        f"请问{random.choice(locations)}有没有{random.choice(subjects)}相关的{random.choice(objects)}？",
        f"我{random.choice(times)}要去{random.choice(locations)}{random.choice(verbs)}{random.choice(subjects)}。",
        f"请问{random.choice(subjects)}的{random.choice(verbs)}结果出来了吗？",
        f"{random.choice(people)}，{random.choice(subjects)}的{random.choice(objects)}需要{random.choice(times)}前完成。",
        f"我已预约了{random.choice(times)}的{random.choice(subjects)}{random.choice(verbs)}，在{random.choice(locations)}。",
    ]
    text = random.choice(patterns)

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

print(f"Generated {count} additional normal general samples")

new_df = pd.DataFrame(new_samples)
combined = pd.concat([df, new_df], ignore_index=True)
combined = combined.drop_duplicates(subset=['text']).reset_index(drop=True)

output_path = r"c:\Users\chuny\Desktop\paper_progarm\data\versions\dataset_20260421_expanded.csv"
combined.to_csv(output_path, index=False, encoding='utf-8')

print(f"\nFinal dataset: {len(combined)}")
print(f"Label distribution: {combined['label'].value_counts().to_dict()}")
print(f"Scenario distribution: {combined['scenario'].value_counts().to_dict()}")
print(f"Ratio: phishing={((combined['label']==1).sum()/len(combined)*100):.1f}%, normal={((combined['label']==0).sum()/len(combined)*100):.1f}%")
