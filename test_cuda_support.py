import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"Nome da GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma'}")