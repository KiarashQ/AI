#!/bin/bash

# --- Get Script Directory ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# --- ADDED: Determine Project Root from Script Directory ---
PROJECT_ROOT=$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )

# --- Call Stop Script ---
echo "Attempting to stop existing server processes via stop_server.sh..."
# Assumes stop_server.sh is also in scripts/ directory
"$SCRIPT_DIR/stop_server.sh"
echo "Finished stopping processes. Waiting a few seconds..."
sleep 3

# --- Define Log Directory and New Log Paths ---
LOG_DIR="$PROJECT_ROOT/logs"
GRPC_SERVER_LOG="$LOG_DIR/grpc_server.log"

# --- Activate Environment ---
echo "Activating Python virtual environment..."
source /home/ubuntu/myenv/bin/activate

# --- *** Load .env file if it exists *** ---
DOTENV_PATH="$PROJECT_ROOT/.env"
if [ -f "$DOTENV_PATH" ]; then
    echo "Loading environment variables from $DOTENV_PATH..."
    eval "$(python -c "import os; from dotenv import dotenv_values; env_vars = dotenv_values('$DOTENV_PATH'); [print(f'export {k}=\"{v.replace('\"','\\\"')}\"') for k, v in env_vars.items() if v is not None and k.isalnum()]")"
    echo ".env file loaded."
else
    echo "No .env file found at $DOTENV_PATH, relying on existing environment variables."
fi

# --- Change to Project Root Directory ---
echo "Changing working directory to $PROJECT_ROOT"
cd "$PROJECT_ROOT" || { echo "Error: Failed to change directory to $PROJECT_ROOT"; exit 1; }

# Add the 'src' directory to PYTHONPATH so Python can find the 'porsit_chatbot' package
export PYTHONPATH="${PROJECT_ROOT}/src"
echo "PYTHONPATH OVERWRITTEN to: $PYTHONPATH"

# --- tart gRPC server ---
echo "Starting gRPC server (module: porsit_chatbot.grpc_server.main)..."
# Run as a module from the project root
nohup python -m porsit_chatbot.grpc_server.main > "$GRPC_SERVER_LOG" 2>&1 &
echo "gRPC server started and logging to $GRPC_SERVER_LOG"

# --- Wait for gRPC server ---
# (Using nc as per original, although ss -tuln | grep ':50051' is often preferred)
echo "Waiting for gRPC server (port 50051) to start..."
for i in {1..10}; do
    if nc -z localhost 50051; then
        echo "gRPC server is up and running on port 50051."
        break
    fi
    echo "Waiting... (attempt $i)"
    sleep 1
done

# --- NGINX Management (Keep if needed) ---
echo "Restarting NGINX..."
sudo systemctl restart nginx
echo "NGINX restarted."

echo "Reloading NGINX..."
sudo systemctl reload nginx
echo "NGINX reloaded."

echo "Testing NGINX configuration..."
sudo nginx -t
if [ $? -ne 0 ]; then
    echo "NGINX configuration test failed. Please check configuration."
    # Consider exiting if NGINX fails: exit 1
fi
echo "NGINX configuration appears valid."

echo "---------------------"
echo "Startup script finished."
echo "---------------------"