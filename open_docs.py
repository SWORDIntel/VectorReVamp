#!/usr/bin/env python3
"""
VectorReVamp Documentation Launcher

This script launches the VectorReVamp HTML documentation in your default web browser.
"""

import os
import sys
import webbrowser
import subprocess
from pathlib import Path


def get_docs_path():
    """Get the path to the documentation index file."""
    script_dir = Path(__file__).parent
    docs_index = script_dir / "docs" / "index.html"

    if not docs_index.exists():
        print(f"Error: Documentation not found at {docs_index}")
        print("Please ensure VectorReVamp is properly installed with documentation.")
        sys.exit(1)

    return docs_index


def start_local_server(port=8000):
    """Start a local HTTP server for the documentation."""
    docs_dir = get_docs_path().parent

    try:
        print(f"Starting local documentation server on http://localhost:{port}")
        print(f"Serving from: {docs_dir}")
        print("Press Ctrl+C to stop the server")

        # Start HTTP server
        subprocess.run([
            sys.executable, "-m", "http.server", str(port)
        ], cwd=docs_dir, check=True)

    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


def open_in_browser(port=8000):
    """Open the documentation in the default web browser."""
    docs_url = f"http://localhost:{port}/index.html"

    try:
        print(f"Opening documentation in browser: {docs_url}")
        webbrowser.open(docs_url)
        return True
    except Exception as e:
        print(f"Error opening browser: {e}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="VectorReVamp Documentation Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python open_docs.py              # Open in browser with local server
  python open_docs.py --server     # Start server only (no browser)
  python open_docs.py --port 8080  # Use custom port
  python open_docs.py --file       # Open HTML file directly (no server)
        """
    )

    parser.add_argument(
        "--server", "-s",
        action="store_true",
        help="Start local server only (don't open browser)"
    )

    parser.add_argument(
        "--file", "-f",
        action="store_true",
        help="Open HTML file directly instead of using server"
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port for local server (default: 8000)"
    )

    args = parser.parse_args()

    if args.file:
        # Open HTML file directly
        docs_path = get_docs_path()
        print(f"Opening documentation file: {docs_path}")
        try:
            webbrowser.open(f"file://{docs_path}")
        except Exception as e:
            print(f"Error opening file: {e}")
            print(f"You can manually open: {docs_path}")
            sys.exit(1)
    else:
        # Use local server
        if args.server:
            # Start server only
            start_local_server(args.port)
        else:
            # Start server in background and open browser
            try:
                # Try to open browser first
                if open_in_browser(args.port):
                    # If browser opened successfully, start server
                    start_local_server(args.port)
                else:
                    print("Failed to open browser, starting server anyway...")
                    start_local_server(args.port)
            except KeyboardInterrupt:
                print("\nServer stopped.")


if __name__ == "__main__":
    main()
