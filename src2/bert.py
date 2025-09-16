import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Verificar disponibilidad de GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

class NewsDataset(Dataset):
    """Dataset personalizado para noticias"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenizar el texto
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTFakeNewsClassifier:
    """Clasificador de fake news usando BERT"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, max_length=512):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device
        
        # Inicializar tokenizer y modelo
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes
        ).to(self.device)
        
        # Métricas de entrenamiento
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def prepare_data(self, df, text_column, label_column, test_size=0.2, val_size=0.1):
        """Preparar y dividir los datos"""
        
        # Obtener textos y etiquetas
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        # Convertir etiquetas a números si son strings
        if isinstance(labels[0], str):
            unique_labels = list(set(labels))
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            labels = [label_to_id[label] for label in labels]
            print(f"Mapeo de etiquetas: {label_to_id}")
        
        # Dividir datos
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, labels, test_size=test_size+val_size, random_state=42, stratify=labels
        )
        
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio, random_state=42, stratify=y_temp
        )
        
        # Crear datasets
        self.train_dataset = NewsDataset(X_train, y_train, self.tokenizer, self.max_length)
        self.val_dataset = NewsDataset(X_val, y_val, self.tokenizer, self.max_length)
        self.test_dataset = NewsDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        print(f"Tamaño del conjunto de entrenamiento: {len(self.train_dataset)}")
        print(f"Tamaño del conjunto de validación: {len(self.val_dataset)}")
        print(f"Tamaño del conjunto de prueba: {len(self.test_dataset)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_loaders(self, batch_size=16):
        """Crear data loaders"""
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
    
    def train(self, epochs=3, learning_rate=2e-5, batch_size=16):
        """Entrenar el modelo"""
        
        # Crear data loaders
        self.create_data_loaders(batch_size)
        
        # Configurar optimizador
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Configurar scheduler
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Función de pérdida
        criterion = nn.CrossEntropyLoss()
        
        print("Iniciando entrenamiento...")
        
        for epoch in range(epochs):
            # Entrenamiento
            self.model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(self.train_loader, desc=f'Época {epoch+1}/{epochs} - Entrenamiento')
            
            for batch in train_pbar:
                # Mover datos a GPU
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validación
            val_loss, val_acc, _, _ = self.evaluate(self.val_loader, split_name="Validación")
            
            # Guardar métricas
            avg_train_loss = total_train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f'Época {epoch+1}/{epochs}:')
            print(f'  Pérdida de entrenamiento: {avg_train_loss:.4f}')
            print(f'  Pérdida de validación: {val_loss:.4f}')
            print(f'  Precisión de validación: {val_acc:.4f}')
            print('-' * 50)
    
    def evaluate(self, data_loader, split_name="Evaluación"):
        """Evaluar el modelo"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            eval_pbar = tqdm(data_loader, desc=f'{split_name}')
            
            for batch in eval_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Obtener predicciones
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calcular métricas
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy, predictions, true_labels
    
    def test_model(self):
        """Evaluar en conjunto de prueba"""
        print("Evaluando en conjunto de prueba...")
        test_loss, test_acc, predictions, true_labels = self.evaluate(
            self.test_loader, "Prueba"
        )
        
        print(f'Pérdida de prueba: {test_loss:.4f}')
        print(f'Precisión de prueba: {test_acc:.4f}')
        
        # Reporte detallado
        print("\nReporte de clasificación:")
        print(classification_report(true_labels, predictions))
        
        # Matriz de confusión
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiquetas Reales')
        plt.xlabel('Predicciones')
        plt.show()
        
        return test_loss, test_acc, predictions, true_labels
    
    def plot_training_history(self):
        """Visualizar historial de entrenamiento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Pérdidas
        ax1.plot(self.train_losses, label='Entrenamiento', marker='o')
        ax1.plot(self.val_losses, label='Validación', marker='s')
        ax1.set_title('Pérdida durante el entrenamiento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        ax1.grid(True)
        
        # Precisión de validación
        ax2.plot(self.val_accuracies, label='Validación', marker='o', color='green')
        ax2.set_title('Precisión de validación')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, texts):
        """Hacer predicciones en nuevos textos"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        for text in texts:
            # Tokenizar
            encoding = self.tokenizer(
                str(text),
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)
                
                predictions.append(pred.item())
                probabilities.append(probs.cpu().numpy()[0])
        
        return predictions, probabilities
    
    def save_model(self, save_path='./bert_fake_news_model'):
        """Guardar el modelo"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Modelo guardado en: {save_path}")
    
    def load_model(self, model_path='./bert_fake_news_model'):
        """Cargar un modelo guardado"""
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        print(f"Modelo cargado desde: {model_path}")

def load_dataset_from_uploaded_file():
    """Carga el dataset desde el archivo subido"""
    try:
        # Leer el archivo usando window.fs
        data = window.fs.readFile('fake_news_dataset.csv', {'encoding': 'utf8'})
        
        # Parsear CSV usando pandas
        from io import StringIO
        df = pd.read_csv(StringIO(data))
        return df, True
    except Exception as e:
        print(f"No se pudo cargar desde archivo subido: {e}")
        return None, False

def create_fake_news_labels(df):
    """Crea etiquetas binarias para clasificación de fake news"""
    # Basándose en el 'subject', crear etiquetas binarias
    # Asumiendo que ciertos subjects indican fake news
    fake_subjects = ['fake', 'conspiracy', 'satire', 'bias', 'junksci', 'hate']
    real_subjects = ['news', 'politics', 'government', 'Middle-east', 'left-news', 'Tech']
    
    def classify_news(subject):
        subject_lower = str(subject).lower()
        if any(fake_word in subject_lower for fake_word in fake_subjects):
            return 1  # Fake
        elif any(real_word in subject_lower for real_word in real_subjects):
            return 0  # Real
        else:
            # Para 'News' genérico, podemos asumir que es real por defecto
            return 0  # Real por defecto
    
    df['binary_label'] = df['subject'].apply(classify_news)
    return df

# Ejemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("🚀 BERT FAKE NEWS CLASSIFIER")
    print("="*60)
    
    # Intentar cargar desde archivo subido primero
    df, loaded_from_file = load_dataset_from_uploaded_file()
    
    if not loaded_from_file:
        DATASET_PATH = '../dataset/dataset/datasets/True.csv' 
        try:
            print(f"Cargando dataset desde: {DATASET_PATH}")
            df = pd.read_csv(DATASET_PATH)
            print(f"✅ Dataset cargado exitosamente desde archivo local")
        except FileNotFoundError:
            print(f"❌ Error: No se pudo encontrar el archivo '{DATASET_PATH}'")
            print("Por favor, asegúrate de que:")
            print("1. El archivo existe en la ruta especificada")
            print("2. La ruta sea correcta")
            print("3. Tengas permisos de lectura en el archivo")
            exit()
    else:
        print("✅ Dataset cargado exitosamente desde archivo subido")
    
    # Mostrar información del dataset
    print(f"\n📊 INFORMACIÓN DEL DATASET:")
    print(f"Shape: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    
    print(f"\n📝 PRIMERAS 3 FILAS:")
    print(df.head(3)[['title', 'subject']])
    
    print(f"\n🏷️ DISTRIBUCIÓN DE SUBJECTS:")
    print(df['subject'].value_counts())
    
    # Crear etiquetas binarias para clasificación
    df = create_fake_news_labels(df)
    
    print(f"\n🎯 DISTRIBUCIÓN DE ETIQUETAS BINARIAS:")
    print("0 = Real News, 1 = Fake News")
    print(df['binary_label'].value_counts())
    
    # Configurar columnas
    TEXT_COLUMN = 'text'  # Columna con el texto de las noticias
    LABEL_COLUMN = 'binary_label'  # Columna con etiquetas binarias (0=real, 1=fake)
    
    try:
        # Verificar que las columnas necesarias existan
        if TEXT_COLUMN not in df.columns:
            print(f"\n⚠️  Columna '{TEXT_COLUMN}' no encontrada. Columnas disponibles: {list(df.columns)}")
            print("Por favor, actualiza TEXT_COLUMN con el nombre correcto.")
            exit()
        
        if LABEL_COLUMN not in df.columns:
            print(f"\n⚠️  Columna '{LABEL_COLUMN}' no encontrada. Columnas disponibles: {list(df.columns)}")
            print("Por favor, actualiza LABEL_COLUMN con el nombre correcto.")
            exit()
        
        # Limpiar datos nulos
        original_len = len(df)
        df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        if len(df) < original_len:
            print(f"⚠️  Se removieron {original_len - len(df)} filas con valores nulos")
        
        print(f"\n✅ DATASET FINAL PARA ENTRENAMIENTO:")
        print(f"Total de muestras: {len(df)}")
        print(f"Distribución final de etiquetas:")
        label_counts = df[LABEL_COLUMN].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            label_name = "Fake News" if label == 1 else "Real News"
            print(f"  {label} ({label_name}): {count} ({percentage:.1f}%)")
        
        # Crear el clasificador
        print("\n" + "="*50)
        print("🤖 INICIALIZANDO CLASIFICADOR BERT...")
        print("="*50)
        classifier = BERTFakeNewsClassifier(model_name='distilbert-base-uncased')
        
        # Preparar datos (esto divide en train/val/test automáticamente)
        print("\n📊 PREPARANDO DATOS...")
        classifier.prepare_data(df, TEXT_COLUMN, LABEL_COLUMN)
        
        # Entrenar el modelo
        print("\n🚀 INICIANDO ENTRENAMIENTO...")
        classifier.train(epochs=3, batch_size=8, learning_rate=2e-5)
        
        # Visualizar entrenamiento
        print("\n📈 MOSTRANDO HISTORIAL DE ENTRENAMIENTO...")
        classifier.plot_training_history()
        
        # Evaluar en conjunto de prueba
        print("\n🧪 EVALUACIÓN EN CONJUNTO DE PRUEBA...")
        classifier.test_model()
        
        # Ejemplo de predicción con algunos textos del dataset
        sample_texts = df[TEXT_COLUMN].head(3).tolist()
        sample_labels = df[LABEL_COLUMN].head(3).tolist()
        
        print("\n🔮 EJEMPLOS DE PREDICCIÓN:")
        predictions, probabilities = classifier.predict(sample_texts)
        
        for i, (text, true_label, pred, prob) in enumerate(zip(sample_texts, sample_labels, predictions, probabilities)):
            pred_label = "FAKE" if pred == 1 else "REAL"
            true_label_name = "FAKE" if true_label == 1 else "REAL"
            confidence = max(prob) * 100
            correct = "✅" if pred == true_label else "❌"
            
            print(f"\nEjemplo {i+1}:")
            print(f"Texto: {text[:100]}...")
            print(f"Real: {true_label_name} | Predicción: {pred_label} | Confianza: {confidence:.1f}% {correct}")
        
        # Guardar el modelo
        print(f"\n💾 GUARDANDO MODELO...")
        classifier.save_model('./bert_fake_news_model')
        
        print("\n" + "="*60)
        print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("="*60)
        print("✅ El modelo ha sido entrenado y guardado")
        print("✅ Puedes usar el modelo para clasificar nuevas noticias")
        print("✅ Los archivos del modelo están en './bert_fake_news_model/'")
        
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        print("Verifica que:")
        print("1. El dataset tenga el formato correcto (CSV)")
        print("2. Las columnas especificadas existan")
        print("3. No haya valores nulos en las columnas principales")
        import traceback
        traceback.print_exc()