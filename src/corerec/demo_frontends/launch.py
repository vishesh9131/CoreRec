#!/usr/bin/env python3
"""
Launch script for CoreRec Demo Frontends

This script starts the FastAPI backend and serves the web frontend.
"""

import sys
import os
import subprocess
import time
import threading
import webbrowser
from pathlib import Path
import argparse


def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting CoreRec Demo Frontends API...")

    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    try:
        import uvicorn
        from backend.api import app

        # Run the FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except ImportError:
        print("❌ Error: FastAPI or uvicorn not installed.")
        print("Please install them with: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        sys.exit(1)


def start_frontend_server():
    """Start a simple HTTP server for the frontend."""
    print("🌐 Starting frontend server...")

    web_dir = Path(__file__).parent / "web"

    if not web_dir.exists():
        print(f"❌ Error: Web directory not found at {web_dir}")
        sys.exit(1)

    os.chdir(web_dir)

    try:
        # Use Python's built-in HTTP server
        subprocess.run(
            [sys.executable, "-m", "http.server", "3000", "--bind", "localhost"], check=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend server: {e}")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ["fastapi", "uvicorn", "pandas", "numpy"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True


def wait_for_server(url, timeout=30):
    """Wait for a server to be available."""
    import requests

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)

    return False


def main():
    parser = argparse.ArgumentParser(description="Launch CoreRec Demo Frontends")
    parser.add_argument(
        "--backend-only", action="store_true", help="Start only the backend API server"
    )
    parser.add_argument(
        "--frontend-only", action="store_true", help="Start only the frontend server"
    )
    parser.add_argument(
        "--port", type=int, default=3000, help="Port for frontend server (default: 3000)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't automatically open browser"
    )

    args = parser.parse_args()

    print("🎵 CoreRec Demo Frontends")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    if args.backend_only:
        start_backend()
    elif args.frontend_only:
        start_frontend_server()
    else:
        # Start both backend and frontend
        print("Starting backend and frontend servers...")

        # Start backend in a separate thread
        backend_thread = threading.Thread(target=start_backend, daemon=True)
        backend_thread.start()

        # Wait a moment for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(3)

        # Check if backend is running
        try:
            import requests

            if wait_for_server("http://localhost:8000"):
                print("✅ Backend is running at http://localhost:8000")
            else:
                print("❌ Backend failed to start")
                sys.exit(1)
        except ImportError:
            print("⚠️  Cannot verify backend status (requests not installed)")
            time.sleep(2)  # Give it more time

        # Start frontend server
        frontend_url = f"http://localhost:{args.port}"
        print(f"✅ Frontend will be available at {frontend_url}")

        if not args.no_browser:
            # Open browser after a delay
            def open_browser():
                time.sleep(2)
                webbrowser.open(frontend_url)

            browser_thread = threading.Thread(target=open_browser, daemon=True)
            browser_thread.start()

        # Start frontend server (this will block)
        try:
            os.chdir(Path(__file__).parent / "web")
            subprocess.run(
                [sys.executable, "-m", "http.server", str(args.port), "--bind", "localhost"]
            )
        except KeyboardInterrupt:
            print("\n🛑 Servers stopped")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
