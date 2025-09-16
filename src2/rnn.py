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
import matplotlib.pyplot as plt
import seaborn as sns

# Descargar recursos de NLTK (ejecutar solo la primera vez)
# nltk.download('punkt')
# nltk.download('stopwords')

class FakeNewsDataset(Dataset):
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
        # Tokenizar y convertir a secuencia de índices
        tokens = word_tokenize(text.lower())
        sequence = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
        
        # Padding o truncamiento
        if len(sequence) < self.max_len:
            sequence += [self.vocab_to_idx['<PAD>']] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
            
        return sequence

class FakeNewsRNN(nn.Module):
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
                              batch_first=True, dropout=dropout, 
                              bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout, 
                             bidirectional=bidirectional)
        
        # Calcular dimensión de salida de RNN
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Capas fully connected
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_output_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # RNN
        if self.rnn_type == 'LSTM':
            rnn_out, (hidden, cell) = self.rnn(embedded)
        else:  # GRU
            rnn_out, hidden = self.rnn(embedded)
        
        # Usar el último output (o concatenar último de ambas direcciones si es bidireccional)
        if self.bidirectional:
            # Concatenar hidden states de ambas direcciones
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Fully connected layers
        out = self.dropout(hidden)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

def preprocess_text(text):
    """Preprocesar texto"""
    if pd.isna(text):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Remover caracteres especiales y números
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remover espacios extra
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def build_vocabulary(texts, min_freq=5):
    """Construir vocabulario a partir de los textos"""
    stop_words = set(stopwords.words('english'))
    
    # Tokenizar todos los textos
    all_tokens = []
    for text in texts:
        tokens = word_tokenize(preprocess_text(text))
        # Filtrar stopwords
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        all_tokens.extend(tokens)
    
    # Contar frecuencias
    token_counts = Counter(all_tokens)
    
    # Crear vocabulario con tokens que aparecen al menos min_freq veces
    vocab = ['<PAD>', '<UNK>'] + [token for token, count in token_counts.items() if count >= min_freq]
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
    
    return vocab_to_idx, vocab

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=0.001):
    """Entrenar el modelo"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 50)
        
        scheduler.step()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, device, class_names):
    """Evaluar el modelo"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, report, cm, all_predictions, all_labels

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Graficar historial de entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """Graficar matriz de confusión"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Configuración de paths y carga de datos
def load_dataset():
    """Cargar dataset desde archivos CSV"""
    texts = []
    labels = []
    
    # Configurar paths de los datasets
    REAL_NEWS_PATH = '../dataset/dataset/datasets/True.csv'
    FAKE_NEWS_PATH = '../dataset/dataset/datasets/Fake.csv'
    
    loaded_from_file = False
    
    if not loaded_from_file:
        try:
            print("Cargando noticias reales...")
            # Cargar noticias reales
            real_df = pd.read_csv(REAL_NEWS_PATH)
            print(f"Noticias reales cargadas: {len(real_df)}")
            
            # Verificar columnas disponibles
            print(f"Columnas en archivo real: {real_df.columns.tolist()}")
            
            # Asumir que tiene columnas 'text' o 'title' para el texto
            if 'text' in real_df.columns:
                real_texts = real_df['text'].dropna().tolist()
            elif 'title' in real_df.columns:
                real_texts = real_df['title'].dropna().tolist()
            else:
                # Si no encuentra columnas esperadas, usar la primera columna de texto
                text_col = real_df.select_dtypes(include=['object']).columns[0]
                real_texts = real_df[text_col].dropna().tolist()
                print(f"Usando columna '{text_col}' como texto")
            
            real_labels = [0] * len(real_texts)  # 0 para noticias reales
            
            print("Cargando noticias falsas...")
            # Cargar noticias falsas
            fake_df = pd.read_csv(FAKE_NEWS_PATH)
            print(f"Noticias falsas cargadas: {len(fake_df)}")
            
            # Verificar columnas disponibles
            print(f"Columnas en archivo fake: {fake_df.columns.tolist()}")
            
            # Asumir que tiene columnas 'text' o 'title' para el texto
            if 'text' in fake_df.columns:
                fake_texts = fake_df['text'].dropna().tolist()
            elif 'title' in fake_df.columns:
                fake_texts = fake_df['title'].dropna().tolist()
            else:
                # Si no encuentra columnas esperadas, usar la primera columna de texto
                text_col = fake_df.select_dtypes(include=['object']).columns[0]
                fake_texts = fake_df[text_col].dropna().tolist()
                print(f"Usando columna '{text_col}' como texto")
            
            fake_labels = [1] * len(fake_texts)  # 1 para noticias falsas
            
            # Combinar datasets
            texts = real_texts + fake_texts
            labels = real_labels + fake_labels
            
            print(f"Dataset total: {len(texts)} noticias")
            print(f"Noticias reales: {len(real_texts)}")
            print(f"Noticias falsas: {len(fake_texts)}")
            
            loaded_from_file = True
            
        except FileNotFoundError as e:
            print(f"Error: No se pudo encontrar el archivo {e}")
            print("Usando datos de ejemplo...")
            loaded_from_file = False
        except Exception as e:
            print(f"Error al cargar el dataset: {e}")
            print("Usando datos de ejemplo...")
            loaded_from_file = False
        
    return texts, labels

# Ejemplo de uso
def main():
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando device: {device}')
    
    # Cargar dataset
    texts, labels = load_dataset()
    
    # Preprocesar textos
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Construir vocabulario
    vocab_to_idx, vocab = build_vocabulary(processed_texts)
    vocab_size = len(vocab)
    print(f'Tamaño del vocabulario: {vocab_size}')
    
    # Dividir datos
    X_train, X_temp, y_train, y_temp = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Crear datasets y dataloaders
    train_dataset = FakeNewsDataset(X_train, y_train, vocab_to_idx)
    val_dataset = FakeNewsDataset(X_val, y_val, vocab_to_idx)
    test_dataset = FakeNewsDataset(X_test, y_test, vocab_to_idx)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Hiperparámetros del modelo
    embed_dim = 100
    hidden_dim = 128
    num_layers = 2
    num_classes = 2
    dropout = 0.3
    rnn_type = 'LSTM'  # o 'GRU'
    
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
    
    print(f'Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros')
    
    # Entrenar modelo
    num_epochs = 20
    learning_rate = 0.001
    
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs, device, learning_rate
    )
    
    # Evaluar en conjunto de prueba
    class_names = ['Real', 'Fake']
    accuracy, report, cm, predictions, true_labels = evaluate_model(
        model, test_loader, device, class_names
    )
    
    print(f'\nAccuracy en test set: {accuracy:.4f}')
    print('\nClassification Report:')
    print(report)
    
    # Graficar resultados
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(cm, class_names)
    
    # Guardar modelo
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
            'bidirectional': True
        }
    }, 'fake_news_rnn_model.pth')
    
    print("Modelo guardado como 'fake_news_rnn_model.pth'")

if __name__ == "__main__":
    main()