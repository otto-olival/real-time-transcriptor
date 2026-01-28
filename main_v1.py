"""
Sistema de transcricao em tempo real
"""

import sounddevice as sd
import numpy as np
import time
import queue
import threading
import torch
import whisper

from config import *
from audio_processing import AudioProcessor
from transcription import Transcriber

print('=='*20)
print("Iniciando transcritor")
print('=='*20)

#fila de comunicacoes entre threads
audio_queue = queue.Queue()
results_queue = queue.Queue()

#buffer pra audio continuo
buffer_audios = np.array([],dtype='float32')

#modelos
print(f'Whisper modelo {WHISPER_MODEL}')
whisper_model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)

print("VAD...")
model_vad, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
    trust_repo=True
)

#processadores
audio_processor = AudioProcessor(model_vad, utils)
transcriber = Transcriber(whisper_model)

def audio_callback(indata, frames, time_info, status):
    """
    Callback chamado continuamente pelo stream de audio
    Ele acumula os audios no buffer e envia os chunks deles pro processamento
    """
    global buffer_audios

    if status:
        print(f'[Stream Status] : {status}')
    
    #add audios no buffer
    buffer_audios = np.concatenate([buffer_audios, indata[:, 0]])

    #com audio suficiente, envia eles pro processamento
    if len(buffer_audios) >= BUFFER_SIZE:
        chunk_samples = int(CHUNK_DURATION*SAMPLE_RATE)
        chunk = buffer_audios[:chunk_samples].copy

        #mantem o overlap pro proximo chunk
        overlap_samples = int(OVERLAP * SAMPLE_RATE)
        buffer_audios = buffer_audios[chunk_samples - overlap_samples:]

        #envia o chunk pra fila
        audio_queue.pu({
            'audio' : chunk,
            'timestamp' : time.time()
        })

chunk_counter = 0
def thread_processamento():
    """
    Thread que processa os chunks de audio
    """
    global chunk_counter

    while True:
        try:
            chunk_data = audio_queue.get()
            chunk_counter += 1

            audio = chunk_data['audio']
            timestamp = chunk_data['timestamp']

            # Mostra volume para debug
            rms = audio_processor.calcular_rms(audio)
            print(f"[{chunk_counter}] Vol: {rms:.4f} ", end="")
            
            # Detecta fala
            if not audio_processor.detectar_fala(audio):
                print("→ 🔇")
                continue

            inicio = time.time()
            results = transcriber.transcrever(audio)
            tempo = time.time() - inicio

            if not results['sucesso']:
                print('[ERROR]')
                continue

            