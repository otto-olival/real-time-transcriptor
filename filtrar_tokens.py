#fui tentar fazer o transcritor pra italiano e alguns tokens especiais acabaram fazendo o modelo alucinar
#entao preciso retirar esses tokens (alucinou pra chines do nada)
import re

def limpeza_de_tokens(text):
    """
    Remove tokens especiais como <|zh|>, <|en|>, etc.
    """
    clean_text = re.sub(r'<\|[^|]+\|>', '', text)

    clean_text = ' '.join(clean_text.split()) #tira espacos extras
    return clean_text

