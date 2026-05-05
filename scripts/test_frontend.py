import urllib.request
try:
    r = urllib.request.urlopen("http://localhost:5173/", timeout=5)
    html = r.read().decode()
    has_app = "app" in html
    print("Frontend accessible:", r.status == 200)
    print("Has app mount:", has_app)
    print("HTML length:", len(html))
except Exception as e:
    print("Frontend test failed:", e)

try:
    r = urllib.request.urlopen("http://localhost:8000/api/v1/system/health", timeout=5)
    import json
    data = json.loads(r.read().decode())
    print("Backend health:", data.get("status"))
    print("Backend version:", data.get("version"))
except Exception as e:
    print("Backend test failed:", e)
