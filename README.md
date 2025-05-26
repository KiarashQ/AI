# Porsit Chatbot Backend Microservices

This repository contains the backend gRPC microservices for the Porsit Chatbot. It includes separate services for text-based AI/ML processing and audio-based speech processing.

## Overview

The project consists of two main gRPC services, each residing in its own directory:

1.  **Text Server (`text_server/`)**:
    *   Exposes a `ModelService` via gRPC, typically on port **50051** (configurable).
    *   Handles Natural Language Understanding (NLU) using:
        *   Local classification models (Intent/Entity) via the `ClassifyAll` method.
        *   LangChain with LLMs (e.g., GPT, Gemini) via the `ClassifyAll_LC` method.
    *   Generates conversational responses using LLMs via the `LLMResponse` method.
    *   Summarizes conversations via the `SummarizeConversation` method.
    *   Loads local ML models (Transformers) and configurations.
    *   Manages API clients for OpenAI and Gemini.

2.  **Audio Server (`audio_server/`)**:
    *   Exposes a `ModelService` via gRPC, typically on port **50052** (configurable).
    *   Manages API clients for OpenAI (e.g., for Whisper, TTS) and Speechmatics.
    *   Transcribes Persian audio to text.
    *   Synthesizes text to speech.

Each service is designed to be run independently, for example, in its own Docker container.

## Repository Structure
.
├── text_server/
│ ├── Dockerfile
│ ├── .gitignore
│ ├── requirements.txt
│ ├── scripts/
│ │ └── generate_protos.sh # (for text_server protos)
│ ├── src/
│ │ └── porsit_chatbot/
│ │ ├── grpc_server/
│ │ │ ├── protos/
│ │ │ │ └── models.proto <-- Text service protos
│ │ │ └── main.py <-- Text gRPC server entry
│ │ └── ... (other modules)
│ └── .env.example
│
├── audio_server/
│ ├── Dockerfile
│ ├── .gitignore
│ ├── requirements.txt
│ ├── scripts/
│ │ └── generate_protos.sh # (for audio_server protos)
│ ├── src/
│ │ └── porsit_chatbot/
│ │ ├── grpc_server/
│ │ │ ├── protos/
│ │ │ │ └── speech_models.proto <-- Audio service protos
│ │ │ └── main.py <-- Audio gRPC server entry
│ │ └── ... (other modules)
│ └── .env.example
│
└── README.md (This file)

## Prerequisites

*   Python 3.10 (or a compatible version)
*   `pip` and `venv` (Python package management)
*   Git (for cloning the repository)
*   Docker and Docker Compose (Recommended for running services)
*   Access to any necessary ML model files (ensure paths are correct within each service's configuration)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Set up each service individually:**

    ---
    ### A. Text Server Setup

    Navigate to the text server directory:
    ```bash
    cd text_server
    ```

    1.  **Create and Activate Virtual Environment (Optional but Recommended):**
        ```bash
        python3 -m venv venv_text
        source venv_text/bin/activate
        ```

    2.  **Install Dependencies:**
        ```bash
        pip install -r requirements.txt
        ```

    3.  **Generate Protobuf Files:**
        *(This step is crucial after cloning or anytime `text_server/src/porsit_chatbot/grpc_server/protos/models.proto` is modified).*
        Run this command from the `text_server/` directory:
        ```bash
        python -m grpc_tools.protoc \
               -I=src \
               --python_out=src \
               --pyi_out=src \
               --grpc_python_out=src \
               src/porsit_chatbot/grpc_server/protos/models.proto
        ```
        Alternatively, if `scripts/generate_protos.sh` is configured for this:
        ```bash
        bash scripts/generate_protos.sh
        ```

    ---
    ### B. Audio Server Setup

    Navigate to the audio server directory:
    ```bash
    cd audio_server # If you were in text_server, do: cd ../audio_server
    ```

    1.  **Create and Activate Virtual Environment (Optional but Recommended):**
        ```bash
        python3 -m venv venv_audio
        source venv_audio/bin/activate
        ```

    2.  **Install Dependencies:**
        ```bash
        pip install -r requirements.txt
        ```

    3.  **Generate Protobuf Files:**
        *(This step is crucial after cloning or anytime `audio_server/src/porsit_chatbot/grpc_server/protos/speech_models.proto` is modified).*
        Run this command from the `audio_server/` directory:
        ```bash
        python -m grpc_tools.protoc \
               -I=src \
               --python_out=src \
               --pyi_out=src \
               --grpc_python_out=src \
               src/porsit_chatbot/grpc_server/protos/speech_models.proto
        ```
        Alternatively, if `scripts/generate_protos.sh` is configured for this:
        ```bash
        bash scripts/generate_protos.sh
        ```
    ---
    *(Remember to deactivate the virtual environment if you switch between server setups or when done: `deactivate`)*

## Configuration

Each service relies on environment variables for configuration and secrets, managed via a `.env` file in its respective directory.

### 1. Text Server Configuration (`text_server/.env`)

*   Navigate to the `text_server/` directory.
*   Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
*   Edit `text_server/.env` with your development/production values:
    ```dotenv
    # text_server/.env (Example)
    OPENAI_API_KEY="sk-YOUR_OPENAI_KEY"
    GEMINI_API_KEY="AIzaSy_YOUR_GEMINI_KEY"
    GRPC_PORT="50051" # Default for text server
    LOGGING_LEVEL="INFO"
    INTENT_MODEL_PATH="/path/to/your/intent_model/checkpoint-xxxx"
    ENTITY_MODEL_PATH="/path/to/your/ner_model/checkpoint-xxxx"
    ID2TAG_PATH="/path/to/your/id2tag.json"
    # GRPC_SERVER_ADDRESS="localhost:50051" # Used if another service calls this one
    ```

### 2. Audio Server Configuration (`audio_server/.env`)

*   Navigate to the `audio_server/` directory.
*   Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
*   Edit `audio_server/.env` with your development/production values:
    ```dotenv
    # audio_server/.env (Example)
    OPENAI_API_KEY="sk-YOUR_OPENAI_KEY" # Can be the same or different from text_server's key
    # GEMINI_API_KEY="AIzaSy_YOUR_GEMINI_KEY" # If used by audio server
    SPEECHMATICS_API_KEY="YOUR_SPEECHMATICS_KEY" # If using Speechmatics
    GRPC_PORT="50052" # Default for audio server
    LOGGING_LEVEL="INFO"
    # SPEECH_GRPC_SERVER_ADDRESS="localhost:50052" # Used if another service calls this one
    ```

**Production Environment Notes:**
For production, environment variables can be injected directly into Docker containers or managed via systemd environment files (e.g., `/etc/porsit-chatbot/text_server.env`, `/etc/porsit-chatbot/audio_server.env`). Ensure sensitive keys are secured.
The `PYTHONPATH` for each service when run outside Docker should be set to its respective `src` directory (e.g., `text_server/src` or `audio_server/src`). Dockerfiles typically handle this with `WORKDIR` and `ENV PYTHONPATH`.

## Running the Services

You can run each service directly for local development, or using Docker (recommended for consistency and deployment).

### 1. Local Development (Directly)

**Text Server:**
```bash
cd text_server
source venv_text/bin/activate # If using a venv
# Ensure .env file is configured
python src/porsit_chatbot/grpc_server/main.py

Audio Server:
cd audio_server
source venv_audio/bin/activate # If using a venv
# Ensure .env file is configured
python src/porsit_chatbot/grpc_server/main.py
```
### 2. Using Docker (Recommended)
Dockerfiles are provided in each service's directory (text_server/Dockerfile, audio_server/Dockerfile).

**Build the Docker images:**
From the repository root:
```Bash
docker build -t porsit-text-server -f text_server/Dockerfile ./text_server
docker build -t porsit-audio-server -f audio_server/Dockerfile ./audio_server
```
**Run the Docker containers:**


**Text Server:**
(Ensure text_server/.env is populated or pass environment variables directly)

```Bash
docker run -d --rm \
  --name porsit-text-service \
  -p 50051:50051 \
  --env-file ./text_server/.env \
  # -v /path/to/local/models:/app/models # If models are outside and need to be mounted
  porsit-text-server

```
(Note: The GRPC_PORT inside the container should match the one in .env. The -p host_port:container_port maps it.)

**Audio Server:**
(Ensure audio_server/.env is populated or pass environment variables directly)
```Bash
docker run -d --rm \
  --name porsit-audio-service \
  -p 50052:50052 \
  --env-file ./audio_server/.env \
  porsit-audio-server
```

You can use Docker Compose to manage both services more easily. Create a docker-compose.yml file in the repository root.
Example docker-compose.yml:

```Yaml
version: '3.8'
services:
  text_server:
    build:
      context: ./text_server
      dockerfile: Dockerfile
    container_name: porsit-text-service
    ports:
      - "50051:50051" # Exposes text server's configured GRPC_PORT
    env_file:
      - ./text_server/.env
    # volumes:
      # - ./text_server/src:/app/src # For development: live code reload
      # - /path/to/local/models:/app/models # If models are outside
    restart: unless-stopped

  audio_server:
    build:
      context: ./audio_server
      dockerfile: Dockerfile
    container_name: porsit-audio-service
    ports:
      - "50052:50052" # Exposes audio server's configured GRPC_PORT
    env_file:
      - ./audio_server/.env
    # volumes:
      # - ./audio_server/src:/app/src # For development: live code reload
    restart: unless-stopped

# To run: docker-compose up -d
# To stop: docker-compose down
```

### 3. **Production (systemd - Example)**
If deploying directly on a host using systemd (without Docker), you would create separate service files for each:
*   porsit-text-grpc.service
*   porsit-audio-grpc.service
Each service file would:
*   Specify its WorkingDirectory (e.g., /path/to/repo/text_server).
*   Source its own environment file (e.g., EnvironmentFile=/etc/porsit-chatbot/text_server.env).
*   Set ExecStart to run its main.py script (e.g., /path/to/venv_text/bin/python src/porsit_chatbot/grpc_server/main.py).
*   Ensure PYTHONPATH is correctly set if not using a virtual environment's python directly (e.g. Environment="PYTHONPATH=/path/to/repo/text_server/src").
Example commands for a porsit-text-grpc.service:
```Bash
sudo systemctl daemon-reload
sudo systemctl enable porsit-text-grpc.service
sudo systemctl start porsit-text-grpc.service
sudo systemctl status porsit-text-grpc.service
sudo journalctl -u porsit-text-grpc.service -f
```
(Repeat for porsit-audio-grpc.service)

# Protobuf Generation Details
If you modify the .proto files, you **must** regenerate the Python gRPC code.
1. Ensure grpcio-tools is installed in the respective virtual environment (or globally if not using venvs).
2. Activate the correct virtual environment for the service whose protos you are changing.

**For Text Server** (text_server/src/porsit_chatbot/grpc_server/protos/models.proto):
Run from the text_server/ directory:
```Bash
python -m grpc_tools.protoc \
       -I=src \
       --python_out=src \
       --pyi_out=src \
       --grpc_python_out=src \
       src/porsit_chatbot/grpc_server/protos/models.proto
```

**For Audio Server** (audio_server/src/porsit_chatbot/grpc_server/protos/speech_models.proto):
Run from the audio_server/ directory:
```Bash
python -m grpc_tools.protoc \
       -I=src \
       --python_out=src \
       --pyi_out=src \
       --grpc_python_out=src \
       src/porsit_chatbot/grpc_server/protos/speech_models.proto
```

# Deployment Notes
*  **Environment Variables:** Crucial for production. Ensure API keys and sensitive data are not hardcoded and are managed securely.
*   **Port Conflicts:** Ensure the GRPC_PORT (for text server) and SPEECH_GRPC_PORT (for audio server, or a similarly distinct name like AUDIO_GRPC_PORT) are unique if running on the same host without containerization. Docker port mapping handles this well.
*   **Resource Allocation:** Monitor CPU, memory, and GPU (if applicable) usage for each service.
*   **TLS/SSL:** For gRPC communication in production, especially over public networks, consider using TLS/SSL.
*   **Logging & Monitoring:** Implement robust logging and monitoring for both services.
*   **Service Discovery:** If these services need to call each other or be called by other microservices, consider a service discovery mechanism.

# License
Just Porsit for now