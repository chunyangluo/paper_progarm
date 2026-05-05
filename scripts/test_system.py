import urllib.request
import json
import time
from pathlib import Path

BASE = "http://localhost:8000"

def api_get(path):
    r = urllib.request.urlopen(f"{BASE}{path}", timeout=15)
    return json.loads(r.read().decode())

def api_post(path, data):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}
    )
    r = urllib.request.urlopen(req, timeout=30)
    return json.loads(r.read().decode())

results = []

def record(test_id, name, passed, detail=""):
    results.append({"id": test_id, "name": name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_id}: {name}")
    if detail:
        print(f"         {detail}")

print("=" * 60)
print("TEST SUITE: Single Detection")
print("=" * 60)

try:
    r = api_post("/api/v1/detection/single", {
        "text": "【支付宝】您的账户存在安全风险，请立即点击 https://alipay-secure-verify.com 进行验证解冻",
        "url": "https://alipay-secure-verify.com",
        "scenario": "sms"
    })
    record("TC-3.1", "Phishing SMS detection", r.get("is_phishing") == True, f"prediction={r.get('prediction')}, confidence={r.get('confidence'):.4f}, model={r.get('model')}")
    record("TC-3.2", "Response has sample_id", bool(r.get("sample_id")), f"sample_id={r.get('sample_id')}")
    record("TC-3.3", "Response has processing_time", r.get("processing_time", 0) > 0, f"time={r.get('processing_time'):.3f}s")
    record("TC-3.4", "Confidence in valid range", 0 <= r.get("confidence", -1) <= 1, f"confidence={r.get('confidence')}")
except Exception as e:
    record("TC-3.1", "Phishing SMS detection", False, str(e))

try:
    r = api_post("/api/v1/detection/single", {
        "text": "明天下午3点在会议室开项目进度汇报会，请准时参加。",
        "scenario": "general"
    })
    record("TC-3.5", "Normal text detection", r.get("is_phishing") == False, f"prediction={r.get('prediction')}, confidence={r.get('confidence'):.4f}")
except Exception as e:
    record("TC-3.5", "Normal text detection", False, str(e))

try:
    r = api_post("/api/v1/detection/single", {
        "text": "尊敬的客户，您的银行账户异常，请登录 http://bank-verify.cn 确认身份",
        "url": "http://bank-verify.cn",
        "scenario": "email"
    })
    record("TC-3.6", "Email scenario detection", r.get("scenario") == "email", f"scenario={r.get('scenario')}")
except Exception as e:
    record("TC-3.6", "Email scenario detection", False, str(e))

try:
    r = api_post("/api/v1/detection/single", {
        "text": "点击链接领取优惠券 https://promo-shop.xyz/free",
        "url": "https://promo-shop.xyz/free",
        "scenario": "link"
    })
    record("TC-3.7", "Link scenario detection", r.get("scenario") == "link", f"scenario={r.get('scenario')}")
except Exception as e:
    record("TC-3.7", "Link scenario detection", False, str(e))

try:
    r = api_post("/api/v1/detection/single", {
        "text": "测试多模态模型",
        "url": "https://test.com",
        "scenario": "general",
        "model_type": "multimodal"
    })
    record("TC-3.8", "Multimodal model selection", r.get("model") == "multimodal", f"model={r.get('model')}")
except Exception as e:
    record("TC-3.8", "Multimodal model selection", False, str(e))

try:
    r1 = api_post("/api/v1/detection/single", {"text": "缓存测试文本", "scenario": "general"})
    r2 = api_post("/api/v1/detection/single", {"text": "缓存测试文本", "scenario": "general", "use_cache": True})
    record("TC-3.9", "Cache mechanism", r2.get("from_cache") == True, f"first={r1.get('from_cache')}, second={r2.get('from_cache')}")
except Exception as e:
    record("TC-3.9", "Cache mechanism", False, str(e))

print()
print("=" * 60)
print("TEST SUITE: Batch Detection")
print("=" * 60)

try:
    r = api_post("/api/v1/detection/batch", {
        "items": [
            {"text": "【微信】您的账号存在风险，点击验证", "scenario": "sms"},
            {"text": "明天开会讨论项目方案", "scenario": "general"},
            {"text": "银行通知：您的账户被冻结，请登录解冻", "scenario": "email"}
        ]
    })
    record("TC-4.1", "Batch detection returns results", len(r.get("results", [])) == 3, f"total={r.get('total')}")
    record("TC-4.2", "Batch counts correct", r.get("total") == 3, f"phishing={r.get('phishing_count')}, normal={r.get('normal_count')}")
    record("TC-4.3", "Batch has avg processing time", r.get("avg_processing_time", 0) > 0, f"avg={r.get('avg_processing_time'):.3f}s")
except Exception as e:
    record("TC-4.1", "Batch detection returns results", False, str(e))

print()
print("=" * 60)
print("TEST SUITE: Alert Center")
print("=" * 60)

try:
    r = api_get("/api/v1/alerts?limit=20")
    record("TC-5.1", "List alerts", isinstance(r, list), f"count={len(r)}")
except Exception as e:
    record("TC-5.1", "List alerts", False, str(e))

try:
    r = api_get("/api/v1/alerts/unhandled-count")
    record("TC-5.2", "Unhandled alert count", "unhandled_count" in r, f"count={r.get('unhandled_count')}")
except Exception as e:
    record("TC-5.2", "Unhandled alert count", False, str(e))

try:
    r = api_get("/api/v1/alerts?unhandled_only=true")
    record("TC-5.3", "Filter unhandled alerts", isinstance(r, list), f"count={len(r)}")
except Exception as e:
    record("TC-5.3", "Filter unhandled alerts", False, str(e))

print()
print("=" * 60)
print("TEST SUITE: Sample Management")
print("=" * 60)

try:
    r = api_post("/api/v1/samples", {
        "text": "测试样本数据",
        "url": "https://test-sample.com",
        "scenario": "general",
        "label": 1,
        "source": "test"
    })
    record("TC-6.1", "Create sample", bool(r.get("sample_id")), f"sample_id={r.get('sample_id')}")
    created_sample_id = r.get("sample_id", "")
except Exception as e:
    record("TC-6.1", "Create sample", False, str(e))
    created_sample_id = ""

try:
    r = api_get("/api/v1/samples?limit=20")
    record("TC-6.2", "List samples", isinstance(r, list), f"count={len(r)}")
except Exception as e:
    record("TC-6.2", "List samples", False, str(e))

if created_sample_id:
    try:
        r = api_get(f"/api/v1/samples/{created_sample_id}")
        record("TC-6.3", "Get sample by ID", r.get("sample_id") == created_sample_id, f"sample_id={r.get('sample_id')}")
    except Exception as e:
        record("TC-6.3", "Get sample by ID", False, str(e))

try:
    r = api_post("/api/v1/samples/batch", {
        "samples": [
            {"text": "批量样本1", "scenario": "sms", "label": 1},
            {"text": "批量样本2", "scenario": "general", "label": 0}
        ]
    })
    record("TC-6.4", "Batch create samples", r.get("created_count", 0) == 2, f"created={r.get('created_count')}")
except Exception as e:
    record("TC-6.4", "Batch create samples", False, str(e))

print()
print("=" * 60)
print("TEST SUITE: Model Management")
print("=" * 60)

try:
    r = api_get("/api/v1/models/info")
    record("TC-7.1", "Get model info", "inference_service" in r, f"models={r.get('inference_service', {}).get('loaded_models')}")
except Exception as e:
    record("TC-7.1", "Get model info", False, str(e))

try:
    r = api_get("/api/v1/models")
    record("TC-7.2", "List model versions", isinstance(r, list), f"count={len(r)}")
except Exception as e:
    record("TC-7.2", "List model versions", False, str(e))

try:
    r = api_get("/api/v1/models/performance")
    record("TC-7.3", "Get model performance", "model_type" in r, f"type={r.get('model_type')}, loaded={r.get('is_loaded')}")
except Exception as e:
    record("TC-7.3", "Get model performance", False, str(e))

try:
    r = api_post("/api/v1/models/set-active/bert_textcnn", {})
    record("TC-7.4", "Set active model", r.get("success") == True, f"active={r.get('active_model')}")
except Exception as e:
    record("TC-7.4", "Set active model", False, str(e))

print()
print("=" * 60)
print("TEST SUITE: Training")
print("=" * 60)

try:
    r = api_get("/api/v1/training/tasks?limit=10")
    record("TC-8.1", "List training tasks", isinstance(r, list), f"count={len(r)}")
except Exception as e:
    record("TC-8.1", "List training tasks", False, str(e))

print()
print("=" * 60)
print("TEST SUITE: System Resources & Backup")
print("=" * 60)

try:
    r = api_get("/api/v1/system/resources")
    record("TC-9.1", "Get resource info", "current" in r, f"monitoring={r.get('monitoring')}")
except Exception as e:
    record("TC-9.1", "Get resource info", False, str(e))

try:
    r = api_get("/api/v1/system/trends?days=7")
    record("TC-9.2", "Get trend data", isinstance(r, list), f"count={len(r)}")
except Exception as e:
    record("TC-9.2", "Get trend data", False, str(e))

try:
    r = api_get("/api/v1/system/backups")
    record("TC-9.3", "List backups", isinstance(r, list), f"count={len(r)}")
except Exception as e:
    record("TC-9.3", "List backups", False, str(e))

print()
print("=" * 60)
print("TEST SUMMARY")
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

_out = Path(__file__).resolve().parent.parent / "docs" / "reports" / "test_results.json"
_out.parent.mkdir(parents=True, exist_ok=True)
with open(_out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {_out}")
