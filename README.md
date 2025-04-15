# Local RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides information using local LLM and vector search.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Internet connection for downloading dependencies and models

### Setup on Linux/macOS
1. Clone or download this repository
2. Open a terminal in the repository directory
3. Download the LLM model file (Qwen2-1.5B-Instruct.Q8_0.gguf) and place it in the `models/model_files` directory
   - You can download it from [https://huggingface.co/TheBloke/Qwen2-1.5B-Instruct-GGUF](https://huggingface.co/TheBloke/Qwen2-1.5B-Instruct-GGUF)

### Setup on Windows
1. Clone or download this repository
2. Download the LLM model file (Qwen2-1.5B-Instruct.Q8_0.gguf) and place it in the `models\model_files` directory
   - You can download it from [https://huggingface.co/TheBloke/Qwen2-1.5B-Instruct-GGUF](https://huggingface.co/TheBloke/Qwen2-1.5B-Instruct-GGUF)

### Manual Setup
If you prefer to set up the environment manually:
1. Create a virtual environment:
   ```
   python -m venv mchatbot
   ```
2. Activate the environment using the environment.yaml file
3. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
4. Create the model directory and download the model:
   ```
   mkdir -p models/model_files
   ```
5. Download the LLM model file (Qwen2-1.5B-Instruct.Q8_0.gguf) and place it in the `models/model_files` directory

### Using Conda Environment
If you prefer to use Conda for environment management:

1. Install Miniconda or Anaconda if you haven't already
   - Download from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. Create a new environment from the provided environment.yml file:
   ```
   conda env create -f environment.yml
   ```

3. Activate the conda environment:
   ```
   conda activate mchatbot
   ```

4. Create the model directory and download the model:
   ```
   mkdir -p models/model_files
   ```

5. Download the LLM model file (Qwen2-1.5B-Instruct.Q8_0.gguf) and place it in the `models/model_files` directory

## Running the Application

1. Activate the virtual environment if not already activated:

2. Run the application:
   ```
   python app.py
   ```
3. Open a web browser and go to:
   ```
   http://localhost:8080
   ```

## Configuration

You can configure the application by modifying the `config.py` file or by setting environment variables.

Key configuration options:
- `LLM_MODEL_NAME`: Name of the LLM model to use
- `PINECONE_API_KEY`: API key for Pinecone
- `PINECONE_ENV`: Pinecone environment to use

## Troubleshooting

If you encounter issues:
1. Make sure all dependencies are installed correctly
2. Verify that the model file is in the correct location
3. Check the logs for error messages
4. If using GPUs, ensure you have appropriate CUDA drivers installed

For detailed error logs, see the `logs/medicalbot.log` file.
