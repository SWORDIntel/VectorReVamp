#!/bin/bash
# VectorReVamp Documentation Launcher
# This script launches the VectorReVamp HTML documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$SCRIPT_DIR/docs"
DOCS_INDEX="$DOCS_DIR/index.html"
PORT="${PORT:-8000}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if documentation exists
if [ ! -f "$DOCS_INDEX" ]; then
    echo -e "${RED}Error: Documentation not found at $DOCS_INDEX${NC}"
    echo "Please ensure VectorReVamp is properly installed with documentation."
    exit 1
fi

show_usage() {
    cat << EOF
VectorReVamp Documentation Launcher

Usage: $0 [OPTIONS]

Options:
    -s, --server     Start local server only (don't open browser)
    -f, --file       Open HTML file directly instead of using server
    -p, --port PORT  Port for local server (default: 8000)
    -h, --help       Show this help message

Examples:
    $0                    # Open in browser with local server
    $0 --server          # Start server only (no browser)
    $0 --port 8080       # Use custom port
    $0 --file            # Open HTML file directly (no server)

EOF
}

open_browser() {
    local url="$1"
    echo -e "${BLUE}Opening documentation in browser: $url${NC}"

    # Try different browser commands
    if command -v xdg-open > /dev/null 2>&1; then
        xdg-open "$url" > /dev/null 2>&1 &
    elif command -v open > /dev/null 2>&1; then
        open "$url" > /dev/null 2>&1 &
    elif command -v start > /dev/null 2>&1; then
        start "$url" > /dev/null 2>&1 &
    else
        echo -e "${YELLOW}Could not automatically open browser.${NC}"
        echo -e "${YELLOW}Please manually open: $url${NC}"
        return 1
    fi
    return 0
}

start_server() {
    local port="$1"
    echo -e "${GREEN}Starting local documentation server on http://localhost:$port${NC}"
    echo -e "${GREEN}Serving from: $DOCS_DIR${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""

    cd "$DOCS_DIR"
    python3 -m http.server "$port"
}

# Parse command line arguments
SERVER_ONLY=false
OPEN_FILE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--server)
            SERVER_ONLY=true
            shift
            ;;
        -f|--file)
            OPEN_FILE=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

if [ "$OPEN_FILE" = true ]; then
    # Open HTML file directly
    echo -e "${BLUE}Opening documentation file: $DOCS_INDEX${NC}"
    if open_browser "file://$DOCS_INDEX"; then
        echo -e "${GREEN}Documentation opened successfully.${NC}"
    else
        echo -e "${YELLOW}You can manually open: $DOCS_INDEX${NC}"
    fi
else
    # Use local server
    if [ "$SERVER_ONLY" = true ]; then
        # Start server only
        start_server "$PORT"
    else
        # Start server in background and open browser
        echo -e "${BLUE}Starting documentation server and opening browser...${NC}"

        # Open browser first
        if open_browser "http://localhost:$PORT/index.html"; then
            echo -e "${GREEN}Browser opened successfully. Starting server...${NC}"
            echo ""
        else
            echo -e "${YELLOW}Failed to open browser, starting server anyway...${NC}"
            echo ""
        fi

        # Start server
        start_server "$PORT"
    fi
fi
