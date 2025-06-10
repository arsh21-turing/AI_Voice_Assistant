# Voice Assistant with PDF Knowledge Base

A voice-controlled assistant that can answer questions about vehicle manuals and other documents using built-in RAG (Retrieval Augmented Generation) capabilities.

## Getting Started

### Clone the Repository
```bash
# Clone the repository
git clone https://github.com/arsh21-turing/AI_Voice_Assistant.git
```

### Setting Up the Environment

1. Create a Virtual Environment
```bash
# For Python 3 (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

2. Install Requirements
```bash
# Install all dependencies
pip install -r requirements.txt
```

3. Set up API Keys
Create a .env file in the project root or set environment variables:

```bash
# Add your API key to .env file
echo "GROQ_API_KEY=your_groq_api_key" > .env
```

## Running the Assistant

### Basic Usage
```bash
# Run the voice assistant in normal mode
python main.py
```

### Process a Specific PDF File
```bash
# Process a specific PDF file (will be added to the knowledge base)
python main.py filename.pdf

# Example
python main.py venue.pdf
```

### Interactive Mode
After processing a PDF, the assistant automatically enters interactive mode where you can:
- Ask questions about the processed document
- Type 'exit' to quit

## Key Features

- **Wake Word Detection**: Activate the assistant by saying "hey assistant"
- **Extended Listening**: 30-second listening window to capture complete questions
- **PDF Knowledge Base**: Automatically extracts and indexes content from PDF manuals
- **RAG System**: Uses FAISS for vector search and retrieves relevant document sections
- **Voice Commands**: Adjust speech rate, volume, and voice with simple commands
- **System Monitoring**: Provides real-time feedback about system status and issues
- **Graceful Degradation**: Falls back to simpler functionality when components fail
- **Interactive Mode**: Test document search capabilities through text input

## Voice Commands

- "speak slower/faster" - Adjust speech rate
- "volume up/down" - Change volume
- "use male/female voice" - Switch voice type
- "repeat that" - Repeat the last response
- "system status" - Check the status of system components

## Troubleshooting

If you encounter issues with audio devices:
- Ensure your microphone is connected and working
- Check that your speakers/headphones are connected
- Run `python -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"` to see available microphones

For dependency issues:
- Make sure you're using Python 3.8 or newer
- Try updating pip: `pip install --upgrade pip`
- Install individual problematic packages: `pip install pyttsx3 SpeechRecognition faiss-cpu`

## License

This project is licensed under the MIT License - see the LICENSE file for details.