import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

class NewsDataset(Dataset):
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
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, max_length=512):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes
        ).to(self.device)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def prepare_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        self.train_dataset = NewsDataset(X_train, y_train, self.tokenizer, self.max_length)
        self.val_dataset = NewsDataset(X_val, y_val, self.tokenizer, self.max_length)
        self.test_dataset = NewsDataset(X_test, y_test, self.tokenizer, self.max_length)
    
    def create_data_loaders(self, batch_size=16):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    
    def train(self, epochs=3, learning_rate=2e-5, batch_size=16):
        self.create_data_loaders(batch_size)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(self.train_loader, desc=f'BERT Epoch {epoch+1}/{epochs}')
            
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            val_loss, val_acc, _, _ = self.evaluate(self.val_loader)
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
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
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy, predictions, true_labels
    
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
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, model_path='./bert_fake_news_model'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)