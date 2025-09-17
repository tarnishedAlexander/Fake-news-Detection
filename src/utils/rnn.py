import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
import pandas as pd

def download_nltk_resources():
    resources_to_download = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    
    for resource_path, resource_name in resources_to_download:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)

download_nltk_resources()

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
        
        tokens = self.text_to_sequence(text)
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def text_to_sequence(self, text):
        tokens = word_tokenize(str(text).lower())
        sequence = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
        
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
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                              bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                             bidirectional=bidirectional)
        
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
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
        embedded = self.embedding(x)
        
        if self.rnn_type == 'LSTM':
            rnn_out, (hidden, cell) = self.rnn(embedded)
        else:
            rnn_out, hidden = self.rnn(embedded)
        
        if self.bidirectional:
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            hidden = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
        else:
            hidden = hidden[-1] if self.rnn_type == 'GRU' else hidden[0][-1]
        
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
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def build_vocabulary(texts, min_freq=3, max_vocab_size=20000):
    stop_words = set(stopwords.words('english'))
    
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing texts"):
        tokens = word_tokenize(preprocess_text(text))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
        all_tokens.extend(tokens)
    
    token_counts = Counter(all_tokens)
    
    most_common = token_counts.most_common(max_vocab_size - 2)
    filtered_tokens = [token for token, count in most_common if count >= min_freq]
    
    vocab = ['<PAD>', '<UNK>'] + filtered_tokens
    vocab_to_idx = {token: idx for idx, token in enumerate(vocab)}
    
    return vocab_to_idx, vocab

class RNNFakeNewsClassifier:
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, 
                 num_classes=2, dropout=0.4, rnn_type='LSTM', bidirectional=True, max_len=200):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.vocab_to_idx = None
        
        self.model = FakeNewsRNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        ).to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def prepare_data(self, X_train, X_val, X_test, y_train, y_val, y_test, vocab_to_idx):
        self.vocab_to_idx = vocab_to_idx
        
        processed_X_train = [preprocess_text(text) for text in X_train]
        processed_X_val = [preprocess_text(text) for text in X_val]
        processed_X_test = [preprocess_text(text) for text in X_test]
        
        self.train_dataset = FakeNewsDataset(processed_X_train, y_train, vocab_to_idx, self.max_len)
        self.val_dataset = FakeNewsDataset(processed_X_val, y_val, vocab_to_idx, self.max_len)
        self.test_dataset = FakeNewsDataset(processed_X_test, y_test, vocab_to_idx, self.max_len)
    
    def create_data_loaders(self, batch_size=32):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    
    def train(self, num_epochs=15, learning_rate=0.001, batch_size=32):
        self.create_data_loaders(batch_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        best_val_acc = 0
        patience_counter = 0
        early_stop_patience = 7
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            
            train_pbar = tqdm(self.train_loader, desc=f'RNN Epoch {epoch+1}/{num_epochs}')
            
            for texts, labels in train_pbar:
                texts, labels = texts.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            val_loss, val_acc, _, _ = self.evaluate(self.val_loader)
            
            train_loss = total_train_loss / len(self.train_loader)
            train_acc = correct_train / total_train
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), './result/best_rnn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
        
        self.model.load_state_dict(torch.load('./result/best_rnn_model.pth'))
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for texts, labels in data_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def test_model(self):
        test_loss, test_acc, predictions, true_labels = self.evaluate(self.test_loader)
        
        report = classification_report(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        
        return test_loss, test_acc, predictions, true_labels, report, cm
    
    def predict(self, texts):
        self.model.eval()
        predictions = []
        probabilities = []
        
        for text in texts:
            processed_text = preprocess_text(text)
            tokens = word_tokenize(processed_text.lower())
            sequence = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
            
            if len(sequence) < self.max_len:
                sequence += [self.vocab_to_idx['<PAD>']] * (self.max_len - len(sequence))
            else:
                sequence = sequence[:self.max_len]
            
            input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities_tensor = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                
                predictions.append(predicted_class)
                probabilities.append(probabilities_tensor.cpu().numpy()[0])
        
        return predictions, probabilities
    
    def save_model(self, save_path='fake_news_rnn_complete.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_to_idx': self.vocab_to_idx,
            'model_config': {
                'max_len': self.max_len
            }
        }, save_path)
    
    def load_model(self, model_path='fake_news_rnn_complete.pth'):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.vocab_to_idx = checkpoint['vocab_to_idx']
        self.max_len = checkpoint['model_config']['max_len']