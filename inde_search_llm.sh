#!/bin/bash
#-------------------------------------------------------------------------------------------------#
#-- Script Configuration --#
#-------------------------------------------------------------------------------------------------#
set -euo pipefail

# --- User-configurable Parameters ---
APP_DIR="$HOME/ai_rag_apps" # Consolidated app directory
GRADIO_PORT=12123
OLLAMA_MODEL="huihui_ai/jan-nano-abliterated:latest"
OLLAMA_EMBEDDING_MODEL="nomic-embed-text"

# --- Source File Definitions ---
# These files are expected to be in the ai_web_search_engine/ directory relative to this script's location
SOURCE_DIR_PREFIX="ai_web_search_engine"
SOURCE_GRADIO_FILE="${SOURCE_DIR_PREFIX}/v2_app_searxng_discover.py"
SOURCE_CORE_FILE="${SOURCE_DIR_PREFIX}/rag_core.py"
SOURCE_CLI_FILE="${SOURCE_DIR_PREFIX}/app_cli_rag.py"

# --- Target File Names in APP_DIR ---
TARGET_GRADIO_FILE_NAME="app_gradio_rag.py"
TARGET_CORE_FILE_NAME="rag_core.py"
TARGET_CLI_FILE_NAME="app_cli_rag.py"

# --- System Dependencies ---
DEPENDENCIES=("python3" "pip" "ollama" "curl" "git")

#-------------------------------------------------------------------------------------------------#
#-- Dependency & Source File Check --#
#-------------------------------------------------------------------------------------------------#
echo "Checking for required dependencies..."
for cmd in "${DEPENDENCIES[@]}"; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: Dependency '$cmd' is not installed." >&2; exit 1;
    fi
done
echo "All dependencies found."

echo "Checking for source Python files..."
if [ ! -f "$SOURCE_GRADIO_FILE" ]; then
    echo "Error: Gradio source file not found at $SOURCE_GRADIO_FILE" >&2; exit 1;
fi
if [ ! -f "$SOURCE_CORE_FILE" ]; then
    echo "Error: Core RAG file not found at $SOURCE_CORE_FILE" >&2; exit 1;
fi
if [ ! -f "$SOURCE_CLI_FILE" ]; then
    echo "Error: CLI source file not found at $SOURCE_CLI_FILE" >&2; exit 1;
fi
echo "All source Python files found."

#-------------------------------------------------------------------------------------------------#
#-- Application Directory Setup --#
#-------------------------------------------------------------------------------------------------#
echo "Creating application directory: $APP_DIR"
mkdir -p "$APP_DIR"

GRADIO_SCRIPT_PATH="$APP_DIR/$TARGET_GRADIO_FILE_NAME"
CORE_SCRIPT_PATH="$APP_DIR/$TARGET_CORE_FILE_NAME"
CLI_SCRIPT_PATH="$APP_DIR/$TARGET_CLI_FILE_NAME"

#-------------------------------------------------------------------------------------------------#
#-- Python Environment Setup --#
#-------------------------------------------------------------------------------------------------#
echo "Installing required Python packages..."
echo "Installing Gradio separately..."
pip install --no-cache-dir -U gradio
if [ $? -ne 0 ]; then echo "Error: Failed to install Gradio." >&2; exit 1; fi

echo "Installing remaining Python packages..."
pip install --no-cache-dir -U \
    requests \
    beautifulsoup4 \
    langchain \
    langchain-community \
    langchain-core \
    langchain-ollama \
    faiss-cpu
    # langchain-text-splitter is part of langchain core/community now
if [ $? -ne 0 ]; then echo "Error: Failed to install remaining Python packages." >&2; exit 1; fi
echo "Python packages installed successfully."

#-------------------------------------------------------------------------------------------------#
#-- Ollama Model Setup --#
#-------------------------------------------------------------------------------------------------#
echo "Pulling Ollama LLM model: $OLLAMA_MODEL..."
ollama pull "$OLLAMA_MODEL"
echo "Pulling Ollama embedding model: $OLLAMA_EMBEDDING_MODEL..."
ollama pull "$OLLAMA_EMBEDDING_MODEL"
echo "Ollama models pulled."

#-------------------------------------------------------------------------------------------------#
#-- Copy and Configure Python Application Scripts --#
#-------------------------------------------------------------------------------------------------#
echo "Copying Python application scripts to $APP_DIR..."
cp "$SOURCE_GRADIO_FILE" "$GRADIO_SCRIPT_PATH"
if [ $? -ne 0 ]; then echo "Error: Failed to copy Gradio script." >&2; exit 1; fi
cp "$SOURCE_CORE_FILE" "$CORE_SCRIPT_PATH"
if [ $? -ne 0 ]; then echo "Error: Failed to copy Core RAG script." >&2; exit 1; fi
cp "$SOURCE_CLI_FILE" "$CLI_SCRIPT_PATH"
if [ $? -ne 0 ]; then echo "Error: Failed to copy CLI script." >&2; exit 1; fi
echo "Python scripts copied."

echo "Configuring Python application scripts..."
# Configure Gradio Port in Gradio script
sed -i "s/^GRADIO_PORT_PY = .*/GRADIO_PORT_PY = $GRADIO_PORT/" "$GRADIO_SCRIPT_PATH"
if [ $? -ne 0 ]; then echo "Error configuring Gradio port in $TARGET_GRADIO_FILE_NAME." >&2; exit 1; fi

# Configure Ollama models in rag_core.py
# Note: Ensure these variable names (OLLAMA_MODEL_PY, OLLAMA_EMBEDDING_MODEL_PY)
# exactly match those in your rag_core.py
sed -i "s|^OLLAMA_MODEL_PY = .*|OLLAMA_MODEL_PY = \"$OLLAMA_MODEL\"|" "$CORE_SCRIPT_PATH"
if [ $? -ne 0 ]; then echo "Error configuring OLLAMA_MODEL_PY in $TARGET_CORE_FILE_NAME." >&2; exit 1; fi
sed -i "s|^OLLAMA_EMBEDDING_MODEL_PY = .*|OLLAMA_EMBEDDING_MODEL_PY = \"$OLLAMA_EMBEDDING_MODEL\"|" "$CORE_SCRIPT_PATH"
if [ $? -ne 0 ]; then echo "Error configuring OLLAMA_EMBEDDING_MODEL_PY in $TARGET_CORE_FILE_NAME." >&2; exit 1; fi

# SHARED_CONTEXT_BASE_DIR is already using os.path.expanduser in rag_core.py, so no sed needed for it.
# Other constants like SEARXNG_URL could be configured here if needed:
# Example: SEARXNG_URL_CONFIG="http://my.searxng.instance:8080"
# sed -i "s|^SEARXNG_URL = .*|SEARXNG_URL = \"$SEARXNG_URL_CONFIG\"|" "$CORE_SCRIPT_PATH"

echo "Python application scripts configured."

#-------------------------------------------------------------------------------------------------#
#-- Desktop Shortcut Creation (Gradio UI) --#
#-------------------------------------------------------------------------------------------------#
DESKTOP_SHORTCUT_GRADIO_PATH="$HOME/Desktop/AI_RAG_App_Gradio.desktop"
echo "Creating Gradio UI desktop shortcut at: $DESKTOP_SHORTCUT_GRADIO_PATH"
cat << EOF > "$DESKTOP_SHORTCUT_GRADIO_PATH"
[Desktop Entry]
Version=1.0
Name=AI RAG App (Gradio UI)
Comment=Run the AI RAG application with Gradio Web UI
Exec=bash -c "cd '$APP_DIR' && python3 '$TARGET_GRADIO_FILE_NAME'"
Icon=utilities-terminal
Terminal=true
Type=Application
Categories=Utility;Application;Network;Internet;
EOF
chmod +x "$DESKTOP_SHORTCUT_GRADIO_PATH"
echo "Gradio UI desktop shortcut created."

#-------------------------------------------------------------------------------------------------#
#-- Final Instructions --#
#-------------------------------------------------------------------------------------------------#
echo ""
echo "✅ Setup Complete! AI RAG Applications"
echo "--------------------------------------------------"
echo "Application files are located in: $APP_DIR"
echo "Shared RAG contexts (saved/loaded) will be in: ~/my_rag_shared_contexts"
echo ""
echo "To run the Gradio Web UI application:"
echo "1. Ensure Ollama service is running."
echo "2. Ensure SearXNG instance is running (if not configured differently in $TARGET_CORE_FILE_NAME)."
echo "3. Open the shortcut: $DESKTOP_SHORTCUT_GRADIO_PATH"
echo "   OR run manually:"
echo "   cd '$APP_DIR'"
echo "   python3 '$TARGET_GRADIO_FILE_NAME'"
echo "   Then open your browser to http://localhost:$GRADIO_PORT"
echo ""
echo "To run the CLI (Command Line Interface) application:"
echo "1. Ensure Ollama service is running."
echo "2. Ensure SearXNG instance is running."
echo "3. Open a new terminal and run:"
echo "   cd '$APP_DIR'"
echo "   python3 '$TARGET_CLI_FILE_NAME'"
echo "--------------------------------------------------"
echo "Review $CORE_SCRIPT_PATH for core configurations (models, SearXNG URL, etc.)."
echo "Review $GRADIO_SCRIPT_PATH for Gradio UI specific configurations (port)."
echo "--------------------------------------------------"
