import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Usar backend sin interfaz gráfica
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, max_length=512):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device
        
        # Inicializar tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
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
        
        # Guardar la figura en lugar de mostrarla
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()  # Cerrar la figura para liberar memoria
        print("✅ Matriz de confusión guardada como 'confusion_matrix.png'")
        
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
        
        # Guardar la figura en lugar de mostrarla
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()  # Cerrar la figura para liberar memoria
        print("✅ Historial de entrenamiento guardado como 'training_history.png'")
    
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
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Modelo cargado desde: {model_path}")

def check_training_results(classifier):
    """Verificar si el modelo está sobreajustado"""
    
    print("\n🔍 ANÁLISIS DE RESULTADOS DE ENTRENAMIENTO:")
    
    if hasattr(classifier, 'val_accuracies') and classifier.val_accuracies:
        final_val_acc = classifier.val_accuracies[-1]
        final_train_loss = classifier.train_losses[-1] if classifier.train_losses else 0
        final_val_loss = classifier.val_losses[-1] if classifier.val_losses else 0
        
        print(f"Precisión final de validación: {final_val_acc:.4f}")
        print(f"Pérdida final de entrenamiento: {final_train_loss:.4f}")
        print(f"Pérdida final de validación: {final_val_loss:.4f}")
        
        # Detectar posible overfitting
        if final_val_acc >= 0.99:
            print("⚠️  ADVERTENCIA: Precisión de validación muy alta (>99%)")
            print("   Esto podría indicar:")
            print("   1. Sobreajuste (overfitting)")
            print("   2. Datos muy fáciles de clasificar")
            print("   3. Filtración de datos entre conjuntos")
            print("   4. Dataset demasiado simple")
            
            print("\n💡 RECOMENDACIONES:")
            print("   - Verificar que los datos de entrenamiento/validación estén bien separados")
            print("   - Considerar usar regularización (dropout, weight decay)")
            print("   - Probar con un dataset más desafiante")
            print("   - Reducir el número de épocas de entrenamiento")
            
        return True
    return False

def create_balanced_dataset(df, max_samples_per_class=None):
    """Crear un dataset balanceado"""
    
    print("\n⚖️  BALANCEANDO DATASET...")
    
    # Contar muestras por clase
    class_counts = df['binary_label'].value_counts()
    print(f"Distribución original:")
    for label, count in class_counts.items():
        label_name = "Fake News" if label == 1 else "Real News"
        print(f"  {label} ({label_name}): {count}")
    
    # Si se especifica un máximo, limitarlo
    if max_samples_per_class:
        min_samples = min(max_samples_per_class, class_counts.min())
    else:
        min_samples = class_counts.min()
    
    print(f"\nUsando {min_samples} muestras por clase...")
    
    # Balancear el dataset
    balanced_dfs = []
    for label in df['binary_label'].unique():
        class_df = df[df['binary_label'] == label].sample(n=min_samples, random_state=42)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"✅ Dataset balanceado creado:")
    print(f"Total de muestras: {len(balanced_df)}")
    balanced_counts = balanced_df['binary_label'].value_counts()
    for label, count in balanced_counts.items():
        label_name = "Fake News" if label == 1 else "Real News"
        print(f"  {label} ({label_name}): {count}")
    
    return balanced_df

def load_datasets_from_files():
    """Carga ambos datasets (True.csv y False.csv/Fake.csv) y los combina"""
    
    # Definir las rutas de los archivos (probamos ambas variaciones)
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
    
    # Cargar cada tipo de archivo y asignar etiquetas
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
                    break  # Si encontramos uno, no necesitamos buscar más
                    
            except Exception as e:
                print(f"   ❌ Error cargando {file_path}: {str(e)}")
        
        if not loaded:
            print(f"   ⚠️  No se encontró ningún archivo para noticias {label_name}")
            print(f"      Rutas buscadas: {file_paths}")
    
    # Verificar que se hayan cargado ambos tipos
    if len(dataframes) == 0:
        print("❌ No se pudo cargar ningún archivo")
        return None, False
    elif len(dataframes) == 1:
        print("⚠️  Solo se cargó un tipo de archivo. Se recomienda tener ambos tipos de noticias")
    
    # Combinar los dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\n📊 DATASET COMBINADO:")
    print(f"Total de muestras: {len(combined_df)}")
    print(f"Distribución por fuente:")
    for source in combined_df['source_file'].unique():
        count = len(combined_df[combined_df['source_file'] == source])
        percentage = (count / len(combined_df)) * 100
        label_name = "Noticias Reales" if source == 'true' else "Noticias Falsas"
        print(f"  {source} ({label_name}): {count} ({percentage:.1f}%)")
    
    return combined_df, True

def load_datasets_from_uploaded_files():
    """Carga datasets desde archivos subidos por el usuario"""
    
    # Lista de posibles nombres de archivos subidos
    possible_files = ['True.csv', 'true.csv', 'False.csv', 'false.csv', 
                     'fake_news_true.csv', 'fake_news_false.csv', 'Fake.csv', 'fake.csv']
    
    dataframes = []
    loaded_files = []
    
    print("🔍 BUSCANDO ARCHIVOS SUBIDOS...")
    
    for filename in possible_files:
        try:
            # Intentar leer el archivo usando window.fs
            data = window.fs.readFile(filename, {'encoding': 'utf8'})
            
            # Parsear CSV usando pandas
            from io import StringIO
            df = pd.read_csv(StringIO(data))
            
            # Determinar la etiqueta basada en el nombre del archivo
            if 'true' in filename.lower():
                df['binary_label'] = 0  # Noticias reales
                source_name = 'true'
            elif 'false' in filename.lower() or 'fake' in filename.lower():
                df['binary_label'] = 1  # Noticias falsas
                source_name = 'false'
            else:
                # Si no se puede determinar, asumir que es un dataset mixto
                print(f"⚠️  No se puede determinar el tipo de {filename}. Se asume dataset mixto.")
                continue
            
            df['source_file'] = source_name
            dataframes.append(df)
            loaded_files.append(filename)
            
            print(f"   ✅ {filename}: {len(df)} muestras de noticias {source_name}")
            
        except Exception as e:
            # Archivo no encontrado o error de lectura (normal)
            continue
    
    if len(dataframes) == 0:
        return None, False
    
    # Combinar los dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"\n📊 ARCHIVOS CARGADOS: {', '.join(loaded_files)}")
    print(f"Total de muestras: {len(combined_df)}")
    
    return combined_df, True

# Ejemplo de uso mejorado
if __name__ == "__main__":
    print("="*60)
    print("🚀 BERT FAKE NEWS CLASSIFIER - VERSIÓN MEJORADA")
    print("="*60)
    
    # Intentar cargar desde archivos subidos primero
    df, loaded_from_uploaded = load_datasets_from_uploaded_files()
    
    if not loaded_from_uploaded:
        # Si no hay archivos subidos, intentar cargar desde rutas locales
        df, loaded_from_local = load_datasets_from_files()
        
        if not loaded_from_local:
            print("❌ Error: No se pudieron cargar los datasets")
            print("Por favor, asegúrate de que:")
            print("1. Los archivos True.csv y False.csv/Fake.csv existen")
            print("2. Las rutas sean correctas")
            print("3. Los archivos tengan el formato CSV correcto")
            exit()
        else:
            print("✅ Datasets cargados desde archivos locales")
    else:
        print("✅ Datasets cargados desde archivos subidos")
    
    # Mostrar información del dataset
    print(f"\n📊 INFORMACIÓN DEL DATASET COMBINADO:")
    print(f"Shape: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    
    # Verificar columnas necesarias
    required_columns = ['text', 'title']
    text_column = None
    
    for col in required_columns:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        print(f"❌ Error: No se encontró columna de texto. Columnas disponibles: {list(df.columns)}")
        print("Se requiere una columna llamada 'text' o 'title'")
        exit()
    
    print(f"✅ Usando columna '{text_column}' para el texto")
    
    # Mostrar ejemplos
    print(f"\n📝 EJEMPLOS DEL DATASET:")
    for i, row in df.head(3).iterrows():
        label_name = "FAKE" if row['binary_label'] == 1 else "REAL"
        source = row.get('source_file', 'unknown')
        text_preview = str(row[text_column])[:100] + "..."
        print(f"Ejemplo {i+1} ({source} - {label_name}): {text_preview}")
    
    # Limpiar datos nulos
    original_len = len(df)
    df = df.dropna(subset=[text_column, 'binary_label'])
    if len(df) < original_len:
        print(f"⚠️  Se removieron {original_len - len(df)} filas con valores nulos")
    
    # Balancear dataset (opcional)
    balance_dataset = True  # Cambiar a False si no quieres balancear
    
    if balance_dataset:
        df = create_balanced_dataset(df, max_samples_per_class=10000)  # Limitar a 10k por clase
    
    print(f"\n✅ DATASET FINAL PARA ENTRENAMIENTO:")
    print(f"Total de muestras: {len(df)}")
    print(f"Distribución final de etiquetas:")
    label_counts = df['binary_label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        label_name = "Fake News" if label == 1 else "Real News"
        print(f"  {label} ({label_name}): {count} ({percentage:.1f}%)")
    
    # Crear el clasificador
    print("\n" + "="*50)
    print("🤖 INICIALIZANDO CLASIFICADOR BERT...")
    print("="*50)
    
    # Usar DistilBERT que es más rápido y eficiente
    classifier = BERTFakeNewsClassifier(model_name='distilbert-base-uncased')
    
    # Preparar datos
    print("\n📊 PREPARANDO DATOS...")
    classifier.prepare_data(df, text_column, 'binary_label')
    
    # Entrenar el modelo con configuración más conservadora
    print("\n🚀 INICIANDO ENTRENAMIENTO...")
    classifier.train(epochs=2, batch_size=16, learning_rate=5e-5)  # Reducir épocas y aumentar learning rate
    
    # Verificar resultados de entrenamiento
    check_training_results(classifier)
    
    # Visualizar entrenamiento
    print("\n📈 GENERANDO HISTORIAL DE ENTRENAMIENTO...")
    classifier.plot_training_history()
    
    # Evaluar en conjunto de prueba
    print("\n🧪 EVALUACIÓN EN CONJUNTO DE PRUEBA...")
    classifier.test_model()
    
    # Ejemplos de predicción
    sample_texts = df[text_column].head(5).tolist()
    sample_labels = df['binary_label'].head(5).tolist()
    
    print("\n🔮 EJEMPLOS DE PREDICCIÓN:")
    predictions, probabilities = classifier.predict(sample_texts)
    
    for i, (text, true_label, pred, prob) in enumerate(zip(sample_texts, sample_labels, predictions, probabilities)):
        pred_label = "FAKE" if pred == 1 else "REAL"
        true_label_name = "FAKE" if true_label == 1 else "REAL"
        confidence = max(prob) * 100
        correct = "✅" if pred == true_label else "❌"
        
        print(f"\nEjemplo {i+1}:")
        print(f"Texto: {text[:150]}...")
        print(f"Real: {true_label_name} | Predicción: {pred_label} | Confianza: {confidence:.1f}% {correct}")
    
    # Guardar el modelo
    print(f"\n💾 GUARDANDO MODELO...")
    classifier.save_model('./bert_fake_news_model')
    
    print("\n" + "="*60)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print("="*60)
    print("✅ El modelo ha sido entrenado con ambos datasets")
    print("✅ Dataset balanceado con noticias reales y falsas")
    print("✅ Modelo guardado en './bert_fake_news_model/'")
    print(f"✅ Gráficas guardadas: training_history.png y confusion_matrix.png")