import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Usar backend sin interfaz gráfica
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Descargar recursos de NLTK automáticamente
def download_nltk_resources():
    """Descargar recursos necesarios de NLTK"""
    resources_to_download = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for resource_path, resource_name in resources_to_download:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Descargando recurso NLTK: {resource_name}")
            nltk.download(resource_name, quiet=True)

# Descargar recursos al inicio
download_nltk_resources()

class FakeNewsDataset(Dataset):
    """Dataset personalizado para noticias con RNN"""
    
    def __init__(self, texts, labels, vocab_to_idx, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab_to_idx = vocab_to_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convertir texto a secuencia de índices
        tokens = self.text_to_sequence(text)
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def text_to_sequence(self, text):
        """Convertir texto a secuencia de índices del vocabulario"""
        tokens = word_tokenize(str(text).lower())
        sequence = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
        
        # Padding o truncamiento
        if len(sequence) < self.max_len:
            sequence += [self.vocab_to_idx['<PAD>']] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
            
        return sequence

class FakeNewsRNN(nn.Module):
    """Modelo RNN para clasificación de fake news"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, 
                 dropout=0.3, rnn_type='LSTM', bidirectional=True):
        super(FakeNewsRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # Capa de embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Capa RNN (LSTM o GRU)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                              bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                             bidirectional=bidirectional)
        
        # Calcular dimensión de salida de RNN
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Capas fully connected con batch normalization
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(rnn_output_dim)
        self.fc1 = nn.Linear(rnn_output_dim, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # RNN
        if self.rnn_type == 'LSTM':
            rnn_out, (hidden, cell) = self.rnn(embedded)
        else:  # GRU
            rnn_out, hidden = self.rnn(embedded)
        
        # Usar el último output válido (considerando bidireccionalidad)
        if self.bidirectional:
            # Concatenar hidden states de ambas direcciones
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
        else:
            hidden = hidden[-1] if self.rnn_type == 'GRU' else hidden[0][-1]
        
        # Fully connected layers con regularización
        out = self.batch_norm1(hidden)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        
        out = self.batch_norm2(out)
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        
        out = self.batch_norm3(out)
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        
        out = self.fc4(out)
        
        return out

def preprocess_text(text):
    """Preprocesar texto para RNN"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remover URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remover caracteres especiales pero mantener algunos signos de puntuación importantes
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    
    # Remover números aislados pero mantener palabras que contienen números
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remover espacios extra
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def build_vocabulary(texts, min_freq=3, max_vocab_size=20000):
    """Construir vocabulario optimizado a partir de los textos"""
    stop_words = set(stopwords.words('english'))
    
    print("Construyendo vocabulario...")
    
    # Tokenizar todos los textos con barra de progreso
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizando textos"):
        tokens = word_tokenize(preprocess_text(text))
        # Filtrar stopwords y tokens muy cortos
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        all_tokens.extend(tokens)
    
    # Contar frecuencias
    token_counts = Counter(all_tokens)
    
    print(f"Tokens únicos encontrados: {len(token_counts)}")
    
    # Crear vocabulario con los tokens más frecuentes
    most_common = token_counts.most_common(max_vocab_size - 2)  # -2 para <PAD> y <UNK>
    filtered_tokens = [token for token, count in most_common if count >= min_freq]
    
    vocab = ['<PAD>', '<UNK>'] + filtered_tokens
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
    
    print(f"Vocabulario final: {len(vocab)} tokens")
    
    return vocab_to_idx, vocab

def load_datasets_from_files():
    """Cargar ambos datasets y combinarlos"""
    
    dataset_paths = {
        'true': [
            '../dataset/dataset/datasets/True.csv',
            '../dataset/dataset/datasets/true.csv'
        ],
        'false': [
            '../dataset/dataset/datasets/False.csv', 
            '../dataset/dataset/datasets/Fake.csv',
            '../dataset/dataset/datasets/false.csv',
            '../dataset/dataset/datasets/fake.csv'
        ]
    }
    
    dataframes = []
    
    print("🔍 CARGANDO DATASETS...")
    
    for label_name, file_paths in dataset_paths.items():
        loaded = False
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    print(f"📄 Cargando {file_path}...")
                    df = pd.read_csv(file_path)
                    
                    # Asignar etiqueta binaria
                    df['binary_label'] = 0 if label_name == 'true' else 1
                    df['source_file'] = label_name
                    
                    print(f"   ✅ Cargado: {len(df)} muestras de noticias {label_name}")
                    dataframes.append(df)
                    loaded = True
                    break
                    
            except Exception as e:
                print(f"   ❌ Error cargando {file_path}: {str(e)}")
        
        if not loaded:
            print(f"   ⚠️  No se encontró ningún archivo para noticias {label_name}")
    
    if len(dataframes) == 0:
        print("❌ No se pudo cargar ningún archivo")
        return None, None, False
    
    # Combinar los dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Detectar columna de texto
    text_column = None
    for col in ['text', 'title']:
        if col in combined_df.columns:
            text_column = col
            break
    
    if text_column is None:
        text_columns = combined_df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            text_column = text_columns[0]
            print(f"Usando columna '{text_column}' como texto")
    
    if text_column is None:
        print("❌ No se encontró columna de texto válida")
        return None, None, False
    
    # Extraer textos y etiquetas
    texts = combined_df[text_column].fillna('').tolist()
    labels = combined_df['binary_label'].tolist()
    
    print(f"\n📊 DATASET COMBINADO:")
    print(f"Total de muestras: {len(texts)}")
    print(f"Columna de texto usada: '{text_column}'")
    
    for source in combined_df['source_file'].unique():
        count = len(combined_df[combined_df['source_file'] == source])
        percentage = (count / len(combined_df)) * 100
        label_name = "Noticias Reales" if source == 'true' else "Noticias Falsas"
        print(f"  {source} ({label_name}): {count} ({percentage:.1f}%)")
    
    return texts, labels, True

def create_balanced_dataset(texts, labels, max_samples_per_class=None):
    """Crear dataset balanceado"""
    
    print("\n⚖️  BALANCEANDO DATASET...")
    
    # Convertir a arrays de numpy para facilitar el manejo
    texts = np.array(texts)
    labels = np.array(labels)
    
    # Contar muestras por clase
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Distribución original:")
    for label, count in zip(unique_labels, counts):
        label_name = "Fake News" if label == 1 else "Real News"
        print(f"  {label} ({label_name}): {count}")
    
    # Determinar número de muestras por clase
    min_samples = np.min(counts)
    if max_samples_per_class:
        min_samples = min(max_samples_per_class, min_samples)
    
    print(f"\nUsando {min_samples} muestras por clase...")
    
    # Balancear el dataset
    balanced_texts = []
    balanced_labels = []
    
    for label in unique_labels:
        label_mask = labels == label
        label_texts = texts[label_mask]
        
        # Seleccionar muestras aleatoriamente
        indices = np.random.choice(len(label_texts), min_samples, replace=False)
        
        balanced_texts.extend(label_texts[indices])
        balanced_labels.extend([label] * min_samples)
    
    print(f"✅ Dataset balanceado creado:")
    print(f"Total de muestras: {len(balanced_texts)}")
    
    # Verificar distribución final
    balanced_labels = np.array(balanced_labels)
    unique_labels, counts = np.unique(balanced_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        label_name = "Fake News" if label == 1 else "Real News"
        print(f"  {label} ({label_name}): {count}")
    
    return balanced_texts, balanced_labels

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=0.001):
    """Entrenar el modelo RNN"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 7
    
    print("Iniciando entrenamiento...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f'Época {epoch+1}/{num_epochs} - Entrenamiento')
        
        for texts, labels in train_pbar:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping para evitar exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validación')
            for texts, labels in val_pbar:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Calcular métricas
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'\nÉpoca [{epoch+1}/{num_epochs}]:')
        print(f'  Pérdida de entrenamiento: {train_loss:.4f}, Precisión: {train_acc:.4f}')
        print(f'  Pérdida de validación: {val_loss:.4f}, Precisión: {val_acc:.4f}')
        print('-' * 60)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Guardar mejor modelo
            torch.save(model.state_dict(), 'best_rnn_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping activado después de {epoch+1} épocas")
                break
    
    # Cargar el mejor modelo
    model.load_state_dict(torch.load('best_rnn_model.pth'))
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, device, class_names):
    """Evaluar el modelo en el conjunto de prueba"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Evaluando')
        for texts, labels in test_pbar:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, report, cm, all_predictions, all_labels, all_probabilities

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Graficar historial de entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(train_losses, label='Entrenamiento', marker='o')
    ax1.plot(val_losses, label='Validación', marker='s')
    ax1.set_title('Pérdida durante el entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(train_accuracies, label='Entrenamiento', marker='o')
    ax2.plot(val_accuracies, label='Validación', marker='s')
    ax2.set_title('Precisión durante el entrenamiento')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('rnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Historial de entrenamiento guardado como 'rnn_training_history.png'")

def plot_confusion_matrix(cm, class_names):
    """Graficar matriz de confusión"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - RNN')
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Reales')
    plt.savefig('rnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Matriz de confusión guardada como 'rnn_confusion_matrix.png'")

def predict_sample_texts(model, vocab_to_idx, sample_texts, device, max_len=200):
    """Hacer predicciones en textos de ejemplo"""
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for text in sample_texts:
            # Preprocesar y tokenizar
            processed_text = preprocess_text(text)
            tokens = word_tokenize(processed_text.lower())
            sequence = [vocab_to_idx.get(token, vocab_to_idx['<UNK>']) for token in tokens]
            
            # Padding o truncamiento
            if len(sequence) < max_len:
                sequence += [vocab_to_idx['<PAD>']] * (max_len - len(sequence))
            else:
                sequence = sequence[:max_len]
            
            # Convertir a tensor
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
            
            # Predecir
            output = model(input_tensor)
            probabilities_tensor = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            
            predictions.append(predicted_class)
            probabilities.append(probabilities_tensor.cpu().numpy()[0])
    
    return predictions, probabilities

def main():
    """Función principal"""
    print("="*70)
    print("🚀 RNN FAKE NEWS CLASSIFIER - VERSIÓN MEJORADA")
    print("="*70)
    
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')
    
    # Cargar datasets
    texts, labels, loaded = load_datasets_from_files()
    
    if not loaded:
        print("❌ Error: No se pudieron cargar los datasets")
        return
    
    # Balancear dataset
    balance_dataset = True
    if balance_dataset:
        texts, labels = create_balanced_dataset(texts, labels, max_samples_per_class=8000)
    
    # Preprocesar textos
    print("\nPreprocesando textos...")
    processed_texts = [preprocess_text(text) for text in tqdm(texts, desc="Preprocesando")]
    
    # Construir vocabulario
    vocab_to_idx, vocab = build_vocabulary(processed_texts, min_freq=3, max_vocab_size=15000)
    vocab_size = len(vocab)
    print(f'Tamaño del vocabulario: {vocab_size}')
    
    # Dividir datos
    print("\nDividiendo datos...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Validación: {len(X_val)} muestras") 
    print(f"Prueba: {len(X_test)} muestras")
    
    # Crear datasets y dataloaders
    max_len = 200
    train_dataset = FakeNewsDataset(X_train, y_train, vocab_to_idx, max_len)
    val_dataset = FakeNewsDataset(X_val, y_val, vocab_to_idx, max_len)
    test_dataset = FakeNewsDataset(X_test, y_test, vocab_to_idx, max_len)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Configuración del modelo
    print("\n" + "="*50)
    print("🤖 CONFIGURANDO MODELO RNN...")
    print("="*50)
    
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    num_classes = 2
    dropout = 0.4
    rnn_type = 'LSTM'  # Cambiar a 'GRU' si prefieres
    
    # Crear modelo
    model = FakeNewsRNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        rnn_type=rnn_type,
        bidirectional=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'Modelo {rnn_type} Bidireccional creado')
    print(f'Parámetros totales: {total_params:,}')
    print(f'Parámetros entrenables: {trainable_params:,}')
    
    # Entrenar modelo
    print("\n🚀 INICIANDO ENTRENAMIENTO...")
    num_epochs = 15
    learning_rate = 0.001
    
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs, device, learning_rate
    )
    
    # Evaluar en conjunto de prueba
    print("\n🧪 EVALUACIÓN EN CONJUNTO DE PRUEBA...")
    class_names = ['Real', 'Fake']
    accuracy, report, cm, predictions, true_labels, probabilities = evaluate_model(
        model, test_loader, device, class_names
    )
    
    print(f'\n✅ Precisión en conjunto de prueba: {accuracy:.4f}')
    print('\n📊 Reporte de clasificación:')
    print(report)
    
    # Graficar resultados
    print("\n📈 GENERANDO GRÁFICAS...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(cm, class_names)
    
    # Ejemplos de predicción
    print("\n🔮 EJEMPLOS DE PREDICCIÓN:")
    sample_texts = X_test[:5] if len(X_test) >= 5 else X_test
    sample_labels = y_test[:5] if len(y_test) >= 5 else y_test
    
    pred_labels, pred_probs = predict_sample_texts(model, vocab_to_idx, sample_texts, device)
    
    for i, (text, true_label, pred_label, prob) in enumerate(zip(sample_texts, sample_labels, pred_labels, pred_probs)):
        pred_name = "FAKE" if pred_label == 1 else "REAL"
        true_name = "FAKE" if true_label == 1 else "REAL"
        confidence = max(prob) * 100
        correct = "✅" if pred_label == true_label else "❌"
        
        print(f"\nEjemplo {i+1}:")
        print(f"Texto: {text[:150]}...")
        print(f"Real: {true_name} | Predicción: {pred_name} | Confianza: {confidence:.1f}% {correct}")
    
    # Guardar modelo completo
    print(f"\n💾 GUARDANDO MODELO...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_to_idx': vocab_to_idx,
        'model_config': {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'dropout': dropout,
            'rnn_type': rnn_type,
            'bidirectional': True,
            'max_len': max_len
        }
    }, 'fake_news_rnn_complete.pth')
    
    print("\n" + "="*70)
    print("🎉 ENTRENAMIENTO RNN COMPLETADO EXITOSAMENTE!")
    print("="*70)
    print("✅ Modelo RNN entrenado con ambos datasets")
    print("✅ Dataset balanceado con noticias reales y falsas")
    print("✅ Modelo guardado como 'fake_news_rnn_complete.pth'")
    print("✅ Gráficas guardadas: rnn_training_history.png y rnn_confusion_matrix.png")
    print(f"✅ Precisión final: {accuracy:.1%}")

if __name__ == "__main__":
    main()