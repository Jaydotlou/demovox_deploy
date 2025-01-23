import gtts
import os
import time
import io
import pyttsx3
import streamlit as st
import speech_recognition as sr
import wave
import logging
import tempfile
import base64
import simpleaudio as sa #Import simpleaudio

from gtts import gTTS
from pydub import AudioSegment
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define the base directory for your project
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Ensure it's set in the .env file.")
os.environ["OPENAI_API_KEY"] = api_key

# Document paths (stored in-memory for Streamlit)
DOC_FILES = ["service_plans.txt", "faq.txt", "support.txt"]

# Load and process documents
def load_and_process_documents(doc_files):
    """Load and process documents into chunks."""
    all_docs = []
    for file in doc_files:
        loader = TextLoader(file)
        documents = loader.load()
        all_docs.extend(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)
    return docs

# Initialize retriever
def initialize_retriever(docs):
    """Initialize vector retriever."""
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

from gtts import gTTS

def text_to_speech_italian(text):
    """Generate WAV output using gTTS and return file path."""
    try:
        # Generate MP3 with gTTS
        tts = gTTS(text, lang="it")
        temp_mp3 = "temp_audio_gtts.mp3"
        tts.save(temp_mp3)

        # Convert to WAV
        temp_wav = "temp_audio_gtts.wav"
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(temp_wav, format="wav")

        # Clean up the MP3 file
        os.remove(temp_mp3)

        return temp_wav
    except Exception as e:
        logger.error(f"Error generating TTS audio: {e}")
        return None

def ensure_pcm_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="wav")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # PCM specs
    audio.export(output_file, format="wav")

# Transcribe audio file
def transcribe_audio_file(file_buffer):
    """Convert Italian audio file (in-memory) to text."""
    recognizer = sr.Recognizer()
    try:
        logger.info(f"Attempting to transcribe audio file: {file_buffer}")
        with sr.AudioFile(file_buffer) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data, language="it-IT")
            logger.info(f"Transcription successful: {transcription}")
            return transcription
    except sr.UnknownValueError:
        logger.error(f"Could not understand the audio in file: {file_buffer}")
        st.error("Non ho capito il tuo input. (I couldn't understand your input.)")
    except sr.RequestError as e:
        logger.error(f"Speech recognition service is unavailable: {e}")
        st.error("Il servizio di riconoscimento vocale non è disponibile. (Service is unavailable.)")
    except Exception as e:
        logger.error(f"General error during transcription: {e}", exc_info=True)
        st.error(f"Errore durante la trascrizione: {e}")
    return None


# Transcribe microphone input
def transcribe_microphone_input():
    """Transcribe speech from the microphone."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Ascoltando... (Listening...)")
        try:
            audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            return recognizer.recognize_google(audio_data, language="it-IT")
        except sr.UnknownValueError:
            st.error("Non ho capito il tuo input. (I couldn't understand your input.)")
            return None
        except sr.RequestError:
            st.error("Il servizio di riconoscimento vocale non è disponibile. (Service is unavailable.)")
            return None

def play_audio_with_html(audio_file_path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(audio_file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        st.error(f"Error playing audio: {e}")

def validate_audio_format(audio_file_path):
    """Validate that the audio is in PCM WAV format."""
    try:
        audio = AudioSegment.from_file(audio_file_path, format="wav")
        if audio.frame_rate == 16000 and audio.channels == 1 and audio.sample_width == 2:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Audio validation failed: {e}")
        return False

# Main function
def main():
    # Load and process documents for Q&A
    docs = load_and_process_documents(DOC_FILES)
    retriever = initialize_retriever(docs)
    llm = ChatOpenAI(temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Streamlit interface
    st.set_page_config(page_title="Italian Chatbot", page_icon=":robot_face:")
    st.title("Chatbot per chiamate simulate in italiano")

    st.markdown("Interagisci con il chatbot caricando un file audio o usando il microfono.")

    # Initialize query variable
    query = None

    # Streamlit interface for input method
    input_method = st.radio("Metodo di input (Input Method):", ["Upload File", "Use Microphone"])
    
    # Handle file uploads
    if input_method == "Upload File":
        audio_files = [
            os.path.join(base_dir, "audio/input/query_it_sample1.wav"),
            os.path.join(base_dir, "audio/input/query_it_sample2.wav"),
            os.path.join(base_dir, "audio/input/query_it_sample3.wav"),
        ]
        selected_file = st.selectbox("Select an audio file for testing:", audio_files)
        if st.button("Process Selected File"):
            if selected_file:
                try:
                    logger.info(f"Processing file: {selected_file}")
                    query = transcribe_audio_file(selected_file)
                except Exception as e:
                    st.error(f"Errore durante l'elaborazione del file audio: {e}")

    # Handle microphone input
    elif input_method == "Use Microphone":
        if st.button("Speak Now"):
            query = transcribe_microphone_input()

# Process query with OpenAI and play the response
    if query:
        st.success(f"Domanda acquisita: {query}")
        with st.spinner("Generazione della risposta... (Generating response...)"):
            try:
                query_in_italian = f"Rispondi in italiano: {query}"
                response = qa_chain.run(query_in_italian)
                st.markdown(f"**Risposta del Chatbot:** {response}")

                # Generate and play audio response
                temp_audio_file_path = text_to_speech_italian(response)
                if temp_audio_file_path:
                    play_audio_with_html(temp_audio_file_path)
                    os.remove(temp_audio_file_path)
                else:
                    st.error("Audio generation failed.")
            except Exception as e:
                st.error(f"Errore durante l'elaborazione della risposta: {e}")
                logger.error(f"Error during response processing: {e}", exc_info=True)

if __name__ == "__main__":
    main()



