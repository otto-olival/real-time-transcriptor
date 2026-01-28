"""
Funcoes pra transcricao e processamento de texto
"""
import re
import whisper 
from scipy.io.wavfile import write
from config import (
    TEMP_FILE,
    SAMPLE_RATE,
    WHISPER_LANGUAGE,
    WHISPER_TEMPERATURE,
    WHISPER_COMPRESSION_RATIO,
    WHISPER_DEVICE,
    WHISPER_LOGPROB_THRESHOLD,
    WHISPER_MODEL,
    WHISPER_NO_SPEECH_THRESHOLD
)

class Transcriber:
    #Gerencia a transcricao de audio p texto
    def __init__(self, model):
        """
        Model : modelo do whisper carregado
        """
        self.model = model
    
    def transcrever(self, audio_data):
        """
        Transcreve audio p texto
        audio_data : Array numpy com amostras dos audios captados

        Retorna:
            dict : {'texto' : str, 'idioma' : str, 'sucesso' : bool}
        """
        try:
            write(TEMP_FILE, SAMPLE_RATE, audio_data)

            #trascreve o audio
            result = self.model.transcribe(
                TEMP_FILE,
                language=WHISPER_LANGUAGE,
                fp16=True,
                temperature = WHISPER_TEMPERATURE,
                compression_ratio_threshold=WHISPER_LOGPROB_THRESHOLD,
                logprob_threshold = WHISPER_LOGPROB_THRESHOLD,
                no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
                condition_on_previous_text=False,
                suppress_tokens=[-1],
                without_timestamps=True
            )

            raw_text = result['text'].strip()

            text = self.limpat_texto(raw_text)

            idioma = result.get('language','unknown')

            return {
                'texto' : text,
                'idioma' : idioma,
                'sucesso' : True
            }
        
        except Exception as e:
            print(f"[TRANSCRIPTION-ERROR] : {e}")
            return {
                'texto' : '',
                'idioma' : '',
                'sucesso' : False
            }
    
    @staticmethod
    #o staticmethod é uma função dentro da classe que não precisa das características da classe pra funcionar
    #não precisa de self, só pega as coisas e processa sem usar o restante da classe pra manipular algum objeto
    #basicamente uma função utilitária dentro da classe
    #é por isso que essa função pode ser colocada DEPOIS de já ter sido incluida no código acima
    #quando o python usa uma classe ele lê o código todo dela antes então não importa onde essa função
    #do staticmethod vai estar definida
    def limpar_texto(text):
        """
        Remove tokens especiais e normaliza o texto
        """
        #remove tokens especiais (</...../>) que sao da lingua transcrita
        clean_text = re.sub(r'<\|[^|]+\|>', '', text)
        
        #remove espacos
        clean_text = ' '.join(clean_text.split())

        return clean_text