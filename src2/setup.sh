#!/bin/bash

echo "🚀 Configurando entorno para BERT Fake News Classifier..."

# Verificar si tenemos GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU NVIDIA detectada"
    GPU_AVAILABLE=true
else
    echo "⚠️  No se detectó GPU NVIDIA, usando CPU"
    GPU_AVAILABLE=false
fi

# Instalar PyTorch
echo "📦 Instalando PyTorch..."
if [ "$GPU_AVAILABLE" = true ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Instalar dependencias principales
echo "📦 Instalando Transformers y dependencias..."
pip install transformers==4.28.1
pip install tokenizers==0.13.3

echo "📦 Instalando librerías de machine learning..."
pip install scikit-learn==1.3.0
pip install pandas==2.0.3
pip install numpy==1.24.3

echo "📦 Instalando librerías de visualización..."
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install tqdm==4.65.0

echo "📦 Instalando aceleradores opcionales..."
pip install accelerate==0.21.0
pip install datasets==2.14.0

# Verificar instalación
echo "🔍 Verificando instalación..."
python -c "
import torch
import transformers
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print('✅ Todas las librerías importadas correctamente')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo "🎉 ¡Configuración completa!"
echo "Ahora puedes ejecutar: python bert.py"
