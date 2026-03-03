import argparse
import os
import signal
import sys

try:
    import uvicorn
except ImportError:
    print("Error: uvicorn not found. Install dependencies first:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Path-Lite development server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    args = parser.parse_args()

    def handle_shutdown(signum, frame):
        print("\nShutting down development server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    print(f"Starting server on {args.host}:{args.port}")
    print("Make sure virtual environment is activated!")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
