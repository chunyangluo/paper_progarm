import os
import sys
import subprocess
import time
import signal
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

processes = []


def install_backend_deps():
    logger.info("Checking backend dependencies...")
    req_file = os.path.join(SRC_DIR, 'api', 'requirements.txt')
    if os.path.exists(req_file):
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', req_file, '-q'],
                check=True, timeout=300
            )
            logger.info("Backend dependencies installed")
        except subprocess.CalledProcessError:
            logger.warning("Some backend dependencies failed to install")


def install_frontend_deps():
    logger.info("Checking frontend dependencies...")
    try:
        subprocess.run(
            ['npm', 'install'],
            cwd=FRONTEND_DIR, check=True, timeout=300,
            shell=True
        )
        logger.info("Frontend dependencies installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Frontend dependencies installation skipped")


def start_backend():
    logger.info("Starting backend service...")
    app_module = "service_layer.app:app"
    env = os.environ.copy()
    env['PYTHONPATH'] = SRC_DIR

    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', app_module,
         '--host', '0.0.0.0', '--port', '8000', '--reload'],
        cwd=SRC_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    processes.append(('backend', proc))
    logger.info(f"Backend service started (PID: {proc.pid})")
    return proc


def start_frontend():
    logger.info("Starting frontend service...")
    proc = subprocess.Popen(
        ['npm', 'run', 'dev'],
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    processes.append(('frontend', proc))
    logger.info(f"Frontend service started (PID: {proc.pid})")
    return proc


def check_health():
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.urlopen('http://localhost:8000/api/v1/system/health', timeout=5)
        if req.status == 200:
            logger.info("Backend health check: OK")
            return True
    except Exception:
        pass
    logger.warning("Backend health check: FAILED")
    return False


def cleanup(signum=None, frame=None):
    logger.info("Shutting down services...")
    for name, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=10)
            logger.info(f"{name} service stopped")
        except Exception:
            proc.kill()
            logger.info(f"{name} service killed")
    processes.clear()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    logger.info("=" * 60)
    logger.info("  Chinese Phishing Text Detection System")
    logger.info("  Starting all services...")
    logger.info("=" * 60)

    install_backend_deps()

    start_backend()

    logger.info("Waiting for backend to start...")
    for i in range(30):
        time.sleep(1)
        if check_health():
            break
    else:
        logger.error("Backend failed to start within 30 seconds")

    start_frontend()

    logger.info("=" * 60)
    logger.info("  All services started!")
    logger.info("  Backend API: http://localhost:8000")
    logger.info("  API Docs: http://localhost:8000/docs")
    logger.info("  Frontend: http://localhost:5173")
    logger.info("  Press Ctrl+C to stop all services")
    logger.info("=" * 60)

    try:
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    logger.error(f"{name} service exited unexpectedly (code: {proc.returncode})")
            time.sleep(5)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
