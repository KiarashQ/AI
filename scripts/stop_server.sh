#!/bin/bash

# Optional: Define log files if needed for upload/rotation later
# LOG_FILES=("$HOME/porsit_chatbot/logs/grpc_server.log")

# Optional: Step 1: Upload logs to MongoDB (Keep commented if not used)
# echo "Uploading logs to MongoDB..."
# Determine project root if log_uploader needs it relative to project
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )
# python "$PROJECT_ROOT/src/porsit_chatbot/utils/log_uploader.py" "${LOG_FILES[@]}" # Adjust path to log_uploader
# echo "Logs uploaded and cleared."

# --- UPDATED: Stop gRPC Server ---
echo "Stopping gRPC server (porsit_chatbot.grpc_server.main)..."
# Use pgrep for a potentially cleaner way to find the PID based on the command line
# The '-f' flag matches against the full command line.
GRPC_SERVER_PID=$(pgrep -f "python -m porsit_chatbot.grpc_server.main")

if [ -n "$GRPC_SERVER_PID" ]; then
  # pgrep might return multiple PIDs if something went wrong, loop through them
  for pid in $GRPC_SERVER_PID; do
      echo "Sending SIGTERM to gRPC server PID $pid..."
      kill "$pid"
  done
  # Add a small wait and potentially SIGKILL if needed after wait
  sleep 2
  GRPC_SERVER_PID_CHECK=$(pgrep -f "python -m porsit_chatbot.grpc_server.main")
  if [ -n "$GRPC_SERVER_PID_CHECK" ]; then
     echo "gRPC server did not stop gracefully, sending SIGKILL..."
     for pid in $GRPC_SERVER_PID_CHECK; do
        kill -9 "$pid"
     done
  fi
  echo "gRPC server stopped."
else
  echo "gRPC server (porsit_chatbot.grpc_server.main) process not found."
fi

echo "---------------------"
echo "Stop script finished."
echo "---------------------"