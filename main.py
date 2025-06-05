import speech_recognition as sr
from speech.recognition import VoiceRecognizer
from speech.synthesis import VoiceSynthesizer
from core.voice_engine import VoiceEngine
from core.assistant_manager import AssistantManager
from rag import QueryProcessor, ContextRetriever, ResponseGenerator, ManualRetriever
from config.settings import config
import time

def main():
    print("Initializing Voice-Powered Car Assistant...")
    
    # Test configuration
    try:
        api_key = config.get('API_SETTINGS', 'GROQ_API_KEY')
        print(f"Configuration loaded successfully. API Key: {api_key[:8]}...")
    except Exception as e:
        print(f"Configuration error: {str(e)}")
        return
    
    # Initialize components
    recognizer = VoiceRecognizer()
    synthesizer = VoiceSynthesizer()
    retriever = ManualRetriever()
    assistant = AssistantManager()
    engine = VoiceEngine(recognizer, synthesizer, retriever, assistant)
    
    print("Voice-Powered Car Assistant is ready. Say 'Hello Car Assistant' to start.")
    
    # Main listening loop
    try:
        while True:
            query = recognizer.listen()
            
            if query:
                print(f"You said: {query}")
                
                if "exit" in query.lower() or "quit" in query.lower():
                    synthesizer.speak("Shutting down. Goodbye!")
                    break
                
                response = engine.process_query(query)
                synthesizer.speak(response)
            
            time.sleep(0.1)  # Small pause to prevent CPU overload
            
    except KeyboardInterrupt:
        print("\nStopping Voice-Powered Car Assistant...")
    
    print("Voice-Powered Car Assistant has shut down.")

if __name__ == "__main__":
    main()