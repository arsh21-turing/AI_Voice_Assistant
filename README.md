# AI Voice Assistant for Car Manuals

## Project Objective
This project provides an intelligent voice assistant specifically designed for car manuals, combining speech recognition, natural language processing, and RAG (Retrieval-Augmented Generation) capabilities to provide context-aware responses about vehicle documentation.
It supports multiple platforms and includes:
- Wake word detection ("hey assistant")
- Extended listening window for complex vehicle-related questions
- Car manual PDF knowledge base with automatic content extraction
- RAG system with FAISS vector search for quick manual lookups
- Voice command controls (speech rate, volume, voice type)
- System monitoring and status feedback
- Graceful degradation for component failures

---
## Code Execution Screenshots
### Conversation 1: Voice Pipeline Initial Setup
Conversation 1 Execution -> https://drive.google.com/file/d/1bxo5m5zd0e8H0pJ_XOljS4P-KmdSqBbw/view?usp=drive_link

Conversation 1 Execution -> https://drive.google.com/file/d/14kczVCIfr8iBoZj0KJrl4XcWjcsLo_3z/view?usp=drive_link

### Conversation 2: PDF Processing and Embedding Pipeline
Conversation 2 Execution -> https://drive.google.com/file/d/15TurdPT3Fjbe3TgHlTj2anmkyZeWZal-/view?usp=drive_link

Conversation 2 Execution -> https://drive.google.com/file/d/1wWG5oRAUYNUWXGC_-BVlMHu4pQQJyXuP/view?usp=drive_link

Conversation 2 Execution -> https://drive.google.com/file/d/1NE_cxdUugihiPo9O7DXKExA2bV6lb4qN/view?usp=drive_link

### Conversation 3: RAG System
Conversation 3 Execution -> https://drive.google.com/file/d/1kqCsE1K1vUOf4lX8z4cBNOw33amn5WRK/view?usp=drive_link

### Conversation 4: Improving RAG and Groq Integration
Conversation 4 Execution -> https://drive.google.com/file/d/1D--RHT78iE6XBkKiMd7ejyk0r55DceVw/view?usp=drive_link

Conversation 4 Execution -> https://drive.google.com/file/d/1wE111VDj73BCxBtB0f2dfwUCFGiIaT5r/view?usp=drive_link

Conversation 4 Execution -> https://drive.google.com/file/d/12n5mtQ9w_DFr_CEdjV2u0a4U5S1QpB4V/view?usp=drive_link

Conversation 4 Execution -> https://drive.google.com/file/d/1vvjjsqUTPUpox36mG8V2_vpHczBbEsOQ/view?usp=drive_link

### Conversation 5: Automotive Queries
Conversation 5 Execution -> https://drive.google.com/file/d/1FKFwhvPT2FrcAMXDvVGyOyZBpyFy-C_0/view?usp=drive_link

### Conversation 6: Conversation Flow
Conversation 6 Execution -> https://drive.google.com/file/d/1q2nGpIqqyvwLa1kTCjRK-U0coYSnHXYa/view?usp=drive_link

### Conversation 7: Advanced Features and Optimization
Conversation 7 Execution -> https://drive.google.com/file/d/1PMk2ms66JDTzo3bRVSL1wiaJPuSZEEA4/view?usp=drive_link

### Conversation 8: Testing and Error Handling
Conversation 8 Execution -> https://drive.google.com/file/d/1WfEtH0u_8HL5iNNFqWTLLvF9d9_6F7qQ/view?usp=drive_link

Conversation 8 Execution -> https://drive.google.com/file/d/1PtMG5PmLiGm5JbZ1Q4lDLz096mY_7pDa/view?usp=drive_link

---
## Unit Test Outputs and Coverage
The following test cases validate critical components of the system across speech recognition, RAG pipeline, and PDF processing.

### RAG Pipeline Tests
- **Test 1**: RAG pipeline processing and retrieval
  Test 1 -> https://drive.google.com/file/d/1At4Kz7Zds74344Ei-tU9oW_wrNQ6I6Dx/view?usp=drive_link

### Embedding Pipeline Tests
- **Test 2**: Embedding generation and management
  Test 2 -> https://drive.google.com/file/d/1Z78wveW-zNCnd2XQspRjrYrPm4BFXVgk/view?usp=drive_link

### Groq Flow Tests
- **Test 3**: Groq API integration and response generation
  Test 3 -> https://drive.google.com/file/d/1W92TQvzdJI-A41vPyzOrZ1xeOU8lVF_J/view?usp=drive_link

### PDF Processing Tests
- **Test 4**: PDF chunking and text extraction
  Test 4 -> https://drive.google.com/file/d/1G9FCn9gyUGwYkrEyihPjLoXyCjbFq9fh/view?usp=drive_link
- **Test 5**: PDF processing and format preservation
  Test 5 -> https://drive.google.com/file/d/18-71zTyh32XgVHDw4DNpzdhKKuAhE5JT/view?usp=drive_link

---
## Project Features Mapped to Components
- **Voice Pipeline**: Wake word detection, speech recognition, and command processing
- **RAG System**: Car manual PDF processing, embedding generation, and document retrieval
- **Voice Commands**: Speech rate, volume, and voice type controls
- **System Monitoring**: Component status and feedback
- **Interactive Mode**: Manual search and query capabilities

---
## Project Features Mapped to Conversations
- **Conversation 1**: Core Voice Pipeline and Terminal Setup - Basic terminal application with SpeechRecognition, voice activity detection, noise filtering, pyttsx3 integration, continuous listening loop, and wake word detection.
- **Conversation 2**: PDF Processing and Embeddings Generation - PDF processing for Venue owner's manual, text chunking with context preservation, embedding generation, local vector database with FAISS, and metadata tagging for manual sections.
- **Conversation 3**: RAG System Implementation - Intelligent retrieval system, context ranking and relevance scoring, query preprocessing for automotive terms, and context assembly system.
- **Conversation 4**: Groq LLM Integration and Prompt Engineering - Groq API integration, automotive-specific prompt templates, context-aware response generation, and error handling with fallback responses.
- **Conversation 5**: Automotive Query Processing - Car-related terminology understanding, synonym mapping for automotive terms, context-aware query expansion, troubleshooting workflows, and maintenance schedules.
- **Conversation 6**: Conversation Flow and User Experience - Natural conversation patterns, confirmation mechanisms, follow-up question support, conversation history, and voice feedback for system status.
- **Conversation 7**: Advanced Features and Optimization - Response time optimization, voice command shortcuts, multiple voice output options, system configuration interface, and local logging system.
- **Conversation 8**: Testing and Error Handling - Comprehensive error handling, automotive query testing, setup documentation, performance monitoring, and system health checks.

---

### Usage:
Run the main application:
```bash
# Basic usage
python main.py

# Process a specific car manual PDF file
python main.py filename.pdf

# Example
python main.py venue.pdf
```

### Voice Commands
- "speak slower/faster" - Adjust speech rate
- "volume up/down" - Change volume
- "use male/female voice" - Switch voice type
- "repeat that" - Repeat the last response
- "system status" - Check the status of system components


# Project Setup

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