import torch
import numpy as np

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
    """
    Detecta se tem alguem falando no audio
        RMS mede o quao alto o som esta a partir da amplitude das ondas que vem do som (RMS eh o root mean square)
    Leva em consideracao a energia do audio e o VAD
        VAD eh um modelo de IA treinado pra diferenciar voz de ruido (rede neural que leva em consideracao coisas como frequencias, padroes temporais de ritmo/pausa e espectro do som, que eh tipo uma impressao digital como um espectro IR etc e tal)
        
    """
    #cada samples de audio eh um numero de -1 a +1 ent eleva ao quadrado pra tirar os valores negativos
    rms = np.sqrt(np.mean(audio_data**2))

    VOLUME_MINIMO = 0.01
    if rms < VOLUME_MINIMO:
        #se o audio for mt baixo nem precisa checar o VAD
        return False
    
    #agora checa o VAD
    try:
        #converte o audio de numpy array pra PyTorch tensor
        audio_tensor = torch.from_numpy(audio_data.flatten()).float()
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            model_vad,
            sampling_rate=sample_rate,
            threshold=0.5,
            min_speech_duration_ms=300,
            min_silence_duration_ms=200,
        )
        
        if len(speech_timestamps) > 0:
            return False
    
    except Exception as e:
        print(f"Erro no vad: {e}")
        return True
