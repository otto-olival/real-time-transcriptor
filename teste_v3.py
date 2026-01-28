import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import queue
import threading
import torch
from filtrar_tokens import limpeza_de_tokens

#configs
SAMPLE_RATE = 16000
CHUNK_DURATION = 3 #isso eh usado no buffer pra que 
OVERLAP = 0.5
TEMP_FILE = "audio_chunk.wav"

#filas pro processamento
audio_queue = queue.Queue()
results_queue = queue.Queue()

#buffer pro audio continuo da stream
buffer_audios = np.array([], dtype='float32')
BUFFER_SIZE = int((CHUNK_DURATION + OVERLAP) * SAMPLE_RATE)

model_name = "small"

print(f"Carregando modelo: {model_name}")
model = whisper.load_model(model_name, device="cuda")

print("Carregando VAD")
model_vad, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
    trust_repo=True
)
(get_speech_timestamps, _, read_audio, *_) = utils

def falou(audio_data, sample_rate=16000):
    #Detecta se tem alguem falando no audio
    try:
        audio_tensor = torch.from_numpy(audio_data.flatten()).float()
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            model_vad,
            sampling_rate=sample_rate,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )
        return len(speech_timestamps) > 0
    
    except Exception as e:
        print(f"Erro no vad: {e}")
        return True
    
#acho que nao precisa limpar o texto por enquanto, caso precise eu faco isso depois

#fazer um callback continuamente pelo stream do audio
def audio_callback(indata, frames, time_info, status):
    """
    A cada bloco de audio essa funcao eh chamada
    -indata tem as amostras de audio mais recentes
    """
    global buffer_audios

    #add novo audio no buffer
    buffer_audios = np.concatenate([buffer_audios, indata[:,0]])

    #quando tiver audio suficiente envia pro processamento
    if len(buffer_audios) >= BUFFER_SIZE:
        #pega o chunk do tamanho definido antes por DURATION E SAMPLE_RATE
        chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
        chunk = buffer_audios[:chunk_samples].copy()

        #mantem overlap pro proximo
        overlap_samples = int(OVERLAP * SAMPLE_RATE)
        buffer_audios = buffer_audios[chunk_samples - overlap_samples:]

        #envia pra a fila de processamento
        audio_queue.put({
            'audio' : chunk,
            'timestamp' : time.time()
        })

#Thread de processamento
chunk_counter = 0 
context_history = []
MAX_CONTEXT = 3

def thread_processamento():
    #processa chunks da fila
    global chunk_counter

    while True:
        try:
            chunk_data = audio_queue.get()
            chunk_counter += 1
            
            audio = chunk_data['audio']
            timestamp = chunk_data['timestamp']

            rms = np.sqrt(np.mean(audio**2))

            print(f"[{chunk_counter}] Vol: {rms:.4f} ", end="")

            # Mostra volume para debug
            # rms = np.sqrt(np.mean(audio**2))
            # print(f"[{chunk_counter}] Vol: {rms:.4f} ", end="")

            #checa se tem fala
            if not falou(audio):
                print("🔇Nada foi falado...🔇")
                continue

            #ainda salva temporariamente, n achei nada pra substituir esse processo
            write(TEMP_FILE, SAMPLE_RATE, audio)

            contexto = ' '.join(context_history[-MAX_CONTEXT:])

            #transcreve
            inicio = time.time()
            result = model.transcribe(
                TEMP_FILE,
                language='pt',  # ou None para auto-detectar
                fp16=True,
                temperature=0.0,
                compression_ratio_threshold=1.8,
                logprob_threshold=-0.5,
                no_speech_threshold=0.7,
                condition_on_previous_text=False,
            )

            tempo = time.time() - inicio

            texto_sujo = result['text'].strip()

            clean_text = limpeza_de_tokens(texto_sujo)

            if clean_text:
                context_history.append(clean_text)
                print(f"✓ ({tempo:.2f}s)")

                results_queue.put({
                    'numero' : chunk_counter,
                    'texto' : clean_text,
                    'tempo' : tempo,
                    'timestamp' : timestamp
                })

            else:
                print("[VAZIO]")
        except Exception as e:
            print(f"[PROCESSING_ERROR] : {e}")

#tudo copiado da IA infelizmente: Thread de exibição
def thread_exibicao():
    """Exibe os resultados"""
    print("💬 Thread de exibição iniciada")
    print("\n" + "=" * 60)
    print("TRANSCRIÇÃO EM TEMPO REAL")
    print("Pressione Ctrl+C para parar")
    print("=" * 60 + "\n")
    
    while True:
        try:
            resultado = results_queue.get()
            
            print(f"\n{'─' * 60}")
            print(f"💬 [{resultado['numero']}] {resultado['texto']}")
            print(f"⏱️  {resultado['tempo']:.2f}s")
            print(f"{'─' * 60}\n")
            
        except Exception as e:
            print(f"[ERROR] Exibição: {e}")

# MAIN
try:
    # Inicia threads de processamento e exibição
    t_processamento = threading.Thread(target=thread_processamento, daemon=True)
    t_exibicao = threading.Thread(target=thread_exibicao, daemon=True)
    
    t_processamento.start()
    t_exibicao.start()
    
    # Inicia o stream de áudio CONTÍNUO    
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1)  # Processa a cada 100ms
    ):        
        # Mantém o programa rodando
        while True:
            time.sleep(0.5)
        
except KeyboardInterrupt:
    print("\n\n🛑 Encerrando...")
    print(f"✓ Chunks processados: {chunk_counter}")
    print(f"✓ Chunks na fila: {audio_queue.qsize()}")
