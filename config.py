"""
Configuracoes do sistema de transcricao
"""

#configs de audio
SAMPLE_RATE = 16000 #Hz
CHUNK_DURATION = 3 #segundos
OVERLAP = 0.5 #segundos
TEMP_FILE = "audio_chunk.wav"

#configs do whisper (modelo de transcricao etc e tal)
WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda"

#configs do VAD (modelo de deteccao de voz)
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 300 #milissegundos
VAD_MIN_SILENCE_MS = 200
RMS_THRESHOLD = 0.01

# Whisper
WHISPER_LANGUAGE = None  # None=auto, 'pt', 'it', 'en', etc.
WHISPER_TEMPERATURE = 0.30
WHISPER_COMPRESSION_RATIO = 1.8
WHISPER_LOGPROB_THRESHOLD = -0.5
WHISPER_NO_SPEECH_THRESHOLD = 0.7

# Anti-alucinação
MAX_REPETICOES = 5  # Máximo de palavras repetidas consecutivas
MAX_TEXTO_PALAVRAS = 100  # Máximo de palavras por transcrição

# Buffer
BUFFER_SIZE = int((CHUNK_DURATION + OVERLAP) * SAMPLE_RATE)

# Threading
BLOCKSIZE = int(SAMPLE_RATE * 0.1)  # Processar a cada 100ms