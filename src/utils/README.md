# Modelos de Clasificación de Noticias Falsas

Este proyecto implementa dos enfoques diferentes para la clasificación de noticias falsas usando deep learning: un modelo basado en **BERT** (Transformer) y un modelo **RNN** personalizado.

## Tabla de Contenidos
- [Modelo BERT](#modelo-bert)
- [Modelo RNN](#modelo-rnn)

## Modelo BERT

### Arquitectura Base
- **Modelo**: DistilBERT-base-uncased (por defecto)
- **Tipo**: Transformer pre-entrenado con fine-tuning
- **Parámetros**: ~66M parámetros (DistilBERT)

### Configuración del Modelo

#### Tokenización
- **Tokenizer**: AutoTokenizer de Transformers
- **Longitud máxima**: 512 tokens
- **Padding**: 'max_length'
- **Truncation**: Habilitado

#### Arquitectura de Clasificación
```
DistilBERT Base → Classification Head
├── Hidden Size: 768
├── Attention Heads: 12
├── Hidden Layers: 6
└── Classification Layer: Linear(768 → 2)
```

#### Configuración de Entrenamiento
- **Optimizador**: AdamW
  - Learning Rate: 2e-5
  - Weight Decay: 0.01 (por defecto)
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: Linear con warmup
  - Warmup Steps: 0
- **Gradient Clipping**: 1.0
- **Batch Size**: 16 (por defecto)
- **Épocas**: 3 (por defecto)

#### Funciones de Activación
- **BERT Layers**: GELU (en capas internas)
- **Classification Head**: Linear (sin activación, logits directos)

### Características Especiales
- **Attention Mechanism**: Multi-head self-attention
- **Transfer Learning**: Fine-tuning de modelo pre-entrenado
- **Regularización**: Dropout implícito en BERT layers

---

## Modelo RNN

### Arquitectura Base
- **Tipo**: LSTM/GRU bidireccional personalizado
- **Enfoque**: Entrenamiento desde cero

### Configuración del Modelo

#### Preprocesamiento de Texto
- **Tokenización**: NLTK word_tokenize
- **Filtros aplicados**:
  - URLs y links removidos
  - Caracteres especiales filtrados
  - Números eliminados
  - Stopwords removidas
  - Conversión a minúsculas
- **Vocabulario**:
  - Tamaño máximo: 20,000 tokens
  - Frecuencia mínima: 3
  - Tokens especiales: `<PAD>`, `<UNK>`

#### Arquitectura de Red
```
Embedding Layer (vocab_size → 128)
    ↓
LSTM/GRU Bidireccional
├── Hidden Dim: 256
├── Num Layers: 2
├── Bidirectional: True
└── Dropout: 0.4 (entre capas)
    ↓
Fully Connected Layers
├── BatchNorm1d(512) → FC(512 → 256) → ReLU → Dropout(0.4)
├── BatchNorm1d(256) → FC(256 → 128) → ReLU → Dropout(0.4)
├── BatchNorm1d(128) → FC(128 → 64) → ReLU
└── FC(64 → 2) [Output]
```

#### Configuración de Entrenamiento
- **Optimizador**: Adam
  - Learning Rate: 0.001
  - Weight Decay: 1e-4
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau
  - Patience: 3
  - Factor: 0.5
- **Gradient Clipping**: 1.0
- **Batch Size**: 32 (por defecto)
- **Épocas**: 15 (por defecto)

#### Funciones de Activación
- **RNN Gates**: Sigmoid y Tanh (LSTM) / Sigmoid (GRU)
- **Dense Layers**: ReLU
- **Output Layer**: Linear (logits)

#### Regularización
- **Dropout**: 0.4 (configurable)
- **Batch Normalization**: En todas las capas densas
- **Early Stopping**: Patience de 7 épocas
- **Gradient Clipping**: Norma máxima de 1.0

### Configuraciones Personalizables
- **RNN Type**: LSTM o GRU
- **Bidireccional**: True/False
- **Embedding Dimension**: 128 (por defecto)
- **Hidden Dimension**: 256 (por defecto)
- **Número de Capas RNN**: 2 (por defecto)
- **Longitud de Secuencia**: 200 tokens (por defecto)

---

## ⚖️ Comparación de Modelos

| Característica | BERT | RNN |
|----------------|------|-----|
| **Arquitectura** | Transformer pre-entrenado | RNN/LSTM bidireccional |
| **Parámetros** | ~66M | ~1-5M (dependiendo config) |
| **Entrenamiento** | Fine-tuning | Desde cero |
| **Memoria GPU** | Alta | Moderada |
| **Tiempo de entrenamiento** | Rápido (pocas épocas) | Moderado (más épocas) |
| **Longitud de secuencia** | 512 tokens | 200 tokens |
| **Preprocesamiento** | Mínimo (tokenizer) | Extensivo (limpieza manual) |
| **Generalización** | Excelente | Buena |
| **Interpretabilidad** | Limitada | Moderada |


## 📊 Métricas de Evaluación

Ambos modelos proporcionan:
- **Accuracy**: Precisión general
- **Classification Report**: Precision, Recall, F1-score por clase
- **Confusion Matrix**: Matriz de confusión
- **Loss Curves**: Pérdida de entrenamiento y validación
- **Probabilidades**: Confianza en las predicciones