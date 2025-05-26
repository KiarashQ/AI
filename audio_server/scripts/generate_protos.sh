#!/bin/bash

# Script to generate Python Protobuf code from .proto files
# Ensures correct paths are used for the src layout.

echo "--- Starting Protobuf Code Generation ---"

# 1. Determine Project Root Directory (assuming script is in scripts/)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
echo "Project Root: $PROJECT_ROOT"

# 2. Define Source and Proto Paths
SRC_DIR="$PROJECT_ROOT/src/porsit_chatbot"
PROTOS_DIR="$SRC_DIR/grpc_server/protos"
OUTPUT_DIR="$PROTOS_DIR" #

# 3. Check if source directory exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Source directory not found at $SRC_DIR"
    exit 1
fi

# 4. Check if proto file exists
if [ ! -f "$PROTO_FILE" ]; then
    echo "Error: Proto file not found at $PROTO_FILE"
    exit 1
fi

# 5. Activate Virtual Environment (Important for finding grpcio-tools)
VENV_PATH="/home/ubuntu/myenv/bin/activate" # Make sure this path is correct
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH"
else
    echo "Warning: Virtual environment not found at $VENV_PATH."
    echo "Ensure grpcio-tools is installed globally or in your PATH if not using a venv."
    # Decide if you want to exit here: exit 1
fi

# 6. Run the protoc command
# Generate for speech_service.proto (the new STT/TTS service)
python -m grpc_tools.protoc \
  -I="$PROTOS_DIR" \
  --python_out="$OUTPUT_DIR" \
  --pyi_out="$OUTPUT_DIR" \
  --grpc_python_out="$OUTPUT_DIR" \
  "$PROTOS_DIR/speech_service.proto"

# 7. Check the exit status of the protoc command
PROTOC_EXIT_CODE=$?
if [ $PROTOC_EXIT_CODE -ne 0 ]; then
    echo "Error: Protobuf generation failed with exit code $PROTOC_EXIT_CODE."
    exit 1 # Exit script with an error status
else
    echo "Protobuf code generated successfully."
    echo "Generated files placed within: $OUTPUT_DIR (relative to proto path)"
fi

echo "--- Protobuf Code Generation Finished ---"
exit 0