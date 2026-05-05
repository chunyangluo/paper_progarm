import torch
from transformers import AutoTokenizer, AutoModel
from core.models.model_definitions import BERTTextCNN

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
bert = AutoModel.from_pretrained("bert-base-chinese")
model = BERTTextCNN(bert)
sd = torch.load(r"c:\Users\chuny\Desktop\paper_progarm\models\bert_textcnn_best.pth", map_location="cpu", weights_only=True)
model.load_state_dict(sd)
model.eval()

texts = [
    "【支付宝】您的账户存在安全风险，请立即点击解冻",
    "明天下午3点开会",
    "银行通知：您的账户被冻结，请登录解冻",
    "恭喜您获得大奖，请点击链接领取",
]
for t in texts:
    inputs = tokenizer(t, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    label = "phishing" if pred == 1 else "normal"
    print(f"Text: {t[:40]}")
    print(f"  Raw output: {out.cpu().numpy()[0].tolist()}")
    print(f"  Probs: normal={probs[0]:.4f}, phishing={probs[1]:.4f}")
    print(f"  Pred: {label}")
    print()
