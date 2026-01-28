import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import os

sample_rate = 16000
duracao = 5
arquivo_temp = "audio_temp.wav"

print("Tentativa de transcriptor em loop")

def load_whisper():
    print("Carregando o modelo")
    return whisper.load_model("small", device="cuda") #o =cuda forca o modelo a usar GPU

model = load_whisper()
round = 1
try:
    while True:
        print("GRAVANDO!")
        audio = sd.rec(int(duracao*sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype='float32')
        sd.wait()

        write(arquivo_temp, sample_rate, audio) #salva temporariamente
        
        inicio_transcricao = time.time()
        resultado = model.transcribe(arquivo_temp, 
                                    language="pt",
                                    fp16=True)  # Usa precisão de 16 bits na GPU
        
        tempo_transcricao = time.time() - inicio_transcricao
        print("\nTranscricao:", resultado["text"].strip())
        print(f"Tempo de transcricao:\n{tempo_transcricao}")
        print(f"📊 Proporção: {tempo_transcricao/duracao:.2f}x o tempo real")
        round = round + 1 

except KeyboardInterrupt:
    print("\n\n🛑 Gravação interrompida pelo usuário")
    print(f"Total de chunks processados: {round - 1}")
    
finally:
    # Limpa arquivo temporário
    if os.path.exists(arquivo_temp):
        os.remove(arquivo_temp)
    print("✓ Arquivo temporário removido")
