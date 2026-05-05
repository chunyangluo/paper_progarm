import urllib.request
import json
import time
from pathlib import Path

BASE = "http://localhost:8000"

def api_get(path):
    try:
        r = urllib.request.urlopen(f"{BASE}{path}", timeout=15)
        return {"status": r.status, "data": json.loads(r.read().decode())}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            body = json.loads(body)
        except:
            pass
        return {"status": e.code, "error": body}
    except Exception as e:
        return {"status": 0, "error": str(e)}

def api_post(path, data):
    try:
        req = urllib.request.Request(
            f"{BASE}{path}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )
        r = urllib.request.urlopen(req, timeout=30)
        return {"status": r.status, "data": json.loads(r.read().decode())}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            body = json.loads(body)
        except:
            pass
        return {"status": e.code, "error": body}
    except Exception as e:
        return {"status": 0, "error": str(e)}

results = []

def record(test_id, name, passed, detail=""):
    results.append({"id": test_id, "name": name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_id}: {name}")
    if detail:
        print(f"         {detail}")

print("=" * 60)
print("EXTENDED TEST: Phishing Detection Accuracy")
print("=" * 60)

phishing_samples = [
    "【银行】尊敬的客户，您的账户已被冻结，请立即登录 http://bank-verify.cn 解冻，否则将被永久封号",
    "紧急通知：您的微信支付存在异常，请点击 https://weixin-safe.com 验证身份，24小时内未验证将永久冻结",
    "恭喜您获得iPhone15大奖！请点击链接 http://prize-claim.xyz 领取，仅限今日！",
    "【快递】您的包裹因地址不详无法派送，请点击 http://express-update.com 更新地址并支付1元保管费",
    "您的QQ账号在异地登录，如非本人操作请点击 https://qq-security.cn 验证",
    "【税务局】您有一笔退税未领取，请登录 http://tax-refund.cn 办理，逾期作废",
    "尊敬的用户，您的信用卡已逾期，请立即还款 https://credit-pay.com 否则影响征信",
    "【社保局】您的社保卡被冻结，点击 http://shebao-update.com 解冻并补录信息",
]

normal_samples = [
    "明天下午3点在会议室开项目进度汇报会，请准时参加",
    "您好，我想咨询一下贵公司的产品价格和配送方式",
    "周末一起去爬山吧，天气预报说周六是晴天",
    "请问图书馆的开放时间是几点到几点？",
    "感谢您的来信，我们会在三个工作日内回复您",
    "今天食堂的午餐有红烧肉和清炒时蔬，味道不错",
    "下周一需要提交季度报告，请各位提前准备",
    "您好，我想预约下周三上午的门诊",
]

phishing_correct = 0
phishing_total = len(phishing_samples)
print("\n--- Phishing Samples ---")
for i, text in enumerate(phishing_samples):
    r = api_post("/api/v1/detection/single", {"text": text, "scenario": "general"})
    is_phish = r.get("data", {}).get("is_phishing", False)
    conf = r.get("data", {}).get("confidence", 0)
    pred = r.get("data", {}).get("prediction", "unknown")
    if is_phish:
        phishing_correct += 1
    print(f"  Sample {i+1}: pred={pred}, conf={conf:.4f}, {'CORRECT' if is_phish else 'WRONG'}")

normal_correct = 0
normal_total = len(normal_samples)
print("\n--- Normal Samples ---")
for i, text in enumerate(normal_samples):
    r = api_post("/api/v1/detection/single", {"text": text, "scenario": "general"})
    is_phish = r.get("data", {}).get("is_phishing", True)
    conf = r.get("data", {}).get("confidence", 0)
    pred = r.get("data", {}).get("prediction", "unknown")
    if not is_phish:
        normal_correct += 1
    print(f"  Sample {i+1}: pred={pred}, conf={conf:.4f}, {'CORRECT' if not is_phish else 'WRONG'}")

phishing_recall = phishing_correct / phishing_total
normal_precision = normal_correct / normal_total
overall_acc = (phishing_correct + normal_correct) / (phishing_total + normal_total)

print(f"\nPhishing Recall: {phishing_correct}/{phishing_total} = {phishing_recall*100:.1f}%")
print(f"Normal Precision: {normal_correct}/{normal_total} = {normal_precision*100:.1f}%")
print(f"Overall Accuracy: {overall_acc*100:.1f}%")

record("TC-E1", f"Phishing recall >= 80%", phishing_recall >= 0.8, f"recall={phishing_recall*100:.1f}%")
record("TC-E2", f"Normal precision >= 80%", normal_precision >= 0.8, f"precision={normal_precision*100:.1f}%")

print()
print("=" * 60)
print("EXTENDED TEST: Error Handling & Edge Cases")
print("=" * 60)

r = api_post("/api/v1/detection/single", {"text": "", "scenario": "general"})
record("TC-E3", "Empty text returns error", r.get("status", 0) != 200, f"status={r.get('status')}")

r = api_post("/api/v1/detection/single", {"text": "a", "scenario": "general"})
record("TC-E4", "Single char text accepted", r.get("status") == 200, f"status={r.get('status')}")

r = api_get("/api/v1/alerts/nonexistent_id")
record("TC-E5", "Nonexistent alert returns 404", r.get("status") == 404, f"status={r.get('status')}")

r = api_get("/api/v1/samples/nonexistent_id")
record("TC-E6", "Nonexistent sample returns 404", r.get("status") == 404, f"status={r.get('status')}")

r = api_get("/api/v1/training/tasks/nonexistent_task")
record("TC-E7", "Nonexistent training task returns 404", r.get("status") == 404, f"status={r.get('status')}")

long_text = "测试" * 1000
r = api_post("/api/v1/detection/single", {"text": long_text, "scenario": "general"})
record("TC-E8", "Long text handled", r.get("status") == 200, f"status={r.get('status')}")

r = api_post("/api/v1/detection/single", {"text": "测试特殊字符 <script>alert(1)</script>", "scenario": "general"})
record("TC-E9", "XSS-like input handled", r.get("status") == 200, f"status={r.get('status')}")

r = api_post("/api/v1/detection/single", {"text": "SQL注入测试 ' OR 1=1 --", "scenario": "general"})
record("TC-E10", "SQL injection-like input handled", r.get("status") == 200, f"status={r.get('status')}")

print()
print("=" * 60)
print("EXTENDED TEST: Data Persistence")
print("=" * 60)

r = api_get("/api/v1/system/stats")
stats_before = r.get("data", {})
record("TC-E11", "Stats reflect previous detections", stats_before.get("total_detections", 0) > 0, f"total={stats_before.get('total_detections')}")

r = api_get("/api/v1/samples?limit=100")
samples_count = len(r.get("data", []))
record("TC-E12", "Previously created samples persist", samples_count > 0, f"count={samples_count}")

print()
print("=" * 60)
print("EXTENDED TEST SUMMARY")
print("=" * 60)
passed = sum(1 for r in results if r["passed"])
failed = sum(1 for r in results if not r["passed"])
total = len(results)
print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
print(f"Pass Rate: {passed/total*100:.1f}%")
if failed > 0:
    print("\nFailed tests:")
    for r in results:
        if not r["passed"]:
            print(f"  - {r['id']}: {r['name']} - {r['detail']}")

_out = Path(__file__).resolve().parent.parent / "docs" / "reports" / "test_results_extended.json"
_out.parent.mkdir(parents=True, exist_ok=True)
with open(_out, "w", encoding="utf-8") as f:
    json.dump({
        "summary": {"total": total, "passed": passed, "failed": failed},
        "phishing_recall": phishing_recall,
        "normal_precision": normal_precision,
        "overall_accuracy": overall_acc,
        "results": results
    }, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {_out}")
