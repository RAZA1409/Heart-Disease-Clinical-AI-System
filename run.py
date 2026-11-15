"""
Multi-Disease Clinical AI Prediction Platform
Application Launcher (Local Development Entry Point)

Runs the complete multi-disease application (Heart Disease, Diabetes, Breast Cancer)
using the configured virtual environment.

Usage:
    python run.py
"""

import os
import sys

def main():
    try:
        from app import app, is_production
    except ImportError as e:
        print("\n" + "=" * 70, file=sys.stderr)
        print("❌ [STARTUP ERROR] Failed to import application dependencies.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("\nPlease ensure your virtual environment is activated and dependencies are installed:", file=sys.stderr)
        print("    .\\.venv\\Scripts\\Activate.ps1", file=sys.stderr)
        print("    python -m pip install -r requirements.txt", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print("\n" + "=" * 70, file=sys.stderr)
        print("❌ [STARTUP ERROR] Required model artifact not found.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70, file=sys.stderr)
        print(f"❌ [STARTUP ERROR] Unexpected error initializing application: {e}", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    # Local development server defaults
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))

    # Enable debug mode for local development unless in production or explicitly disabled
    flask_debug_env = os.getenv("FLASK_DEBUG")
    if flask_debug_env is not None:
        debug = flask_debug_env.lower() in ("true", "1")
    else:
        debug = not is_production

    print("\n" + "=" * 70)
    print("🏥 Multi-Disease Clinical AI Prediction Platform")
    print("=" * 70)
    print(f"📍 Local Server URL   : http://{host}:{port}/login")
    print(f"⚙️  Execution Mode    : {'Production' if is_production else 'Local Development'}")
    print(f"🐞 Debug Mode         : {'Enabled' if debug else 'Disabled'}")
    print("🫀 Modules Active     : Heart Disease | Diabetes | Breast Cancer")
    print("=" * 70)
    print("Press Ctrl+C to stop the development server.\n")

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n[INFO] Development server stopped by user.")
    except Exception as e:
        print(f"\n❌ [SERVER ERROR] Error running development server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

