A Retrieval-Augmented Generation (RAG) chatbot that provides information using local LLM and vector search.

## Prerequisites:
Python 3.8 or higher. Internet connection for downloading dependencies and models. Setup on macOS

## Using Conda Environment:
1. Install Miniconda or Anaconda if you haven't already

2. Download from https://docs.conda.io/en/latest/miniconda.html

3. Clone or download this repository or open the project folder in vs code

4. Open a terminal in the repository director

5. Create a new environment from the provided environment.yml file: conda env create -f environment.yml

6. Download the LLM model file (Qwen2-1.5B-Instruct.Q8_0.gguf) and place it in the `models/model_files` directory. You can download it from <ins>https://huggingface.co/TheBloke/Qwen2-1.5B-Instruct-GGUF</ins>
    
7.  Install the requirements: pip install -r requirements.txt

## Running the Application:
1.  Activate the virtual environment if not already activated: conda activate mchatbot
2.  Run the application: python app.py
3.  Open a web browser and go to: http://localhost:8080

## Key configuration options:
- LLM_MODEL_NAME: Name of the LLM model to use
- PINECONE_API_KEY: API key for Pinecone
- PINECONE_ENV: Pinecone environment to use

## Troubleshooting
If you encounter issues:

Make sure all dependencies are installed correctly
Verify that the model file is in the correct location
Check the logs for error messages

For detailed error logs, see the logs/medicalbot.log file.
