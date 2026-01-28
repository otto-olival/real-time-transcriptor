import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time

#configs da solucao
sample_rate = 16000 #16KHz de frequencia pras amostras
duration = 10 #10 segundos de gravacao de audio
temp_file = "audio_temp.wav" #pra armazenar temporariamente a gravacao

print("=="*10)
print("Trasncritor simples")
print("Carregando modelo do whisper")

time_model = time.time()
#carrega os modelos do whisper
model = whisper.load_model("small") #small e medium funcionaram bem em pt-br
finish_time_model = time.time()

print(f"Gravando por {duration} segundos... FALE AGORA!")

audio = sd.rec(int(duration * sample_rate),
               samplerate=sample_rate,
               channels=1, #mono canal de audio
               dtype='float32'
               )
sd.wait() #espera a gravacao acabar

print("Gravacao finalizada. Iniciando processamento")

#salva no arquivo temporario
write(temp_file, sample_rate, audio)

resultado = model.transcribe(temp_file, language="pt")
print("\n--- TRANSCRIÇÃO ---")
print(resultado["text"])
print("\n--- IDIOMA DETECTADO ---")
print(resultado["language"])
time_transcribe = time.time()
total_time_model = finish_time_model - time_model
total_time_transcribe = time_transcribe - time_model
print(f"Tempo pra carregar modelo:\n{total_time_model}\nTempo para trascrever fala de {duration} segundos:\n{total_time_transcribe}")
