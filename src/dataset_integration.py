import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import requests
import os
import json
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from collections import Counter
import torchvision.transforms as transforms
from transformers import BertTokenizer
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    """Text preprocessing utilities"""
    def __init__(self, vocab_size=10000, max_seq_len=128):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_built = False
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase and remove extra whitespace
        text = ' '.join(text.lower().split())
        
        return text
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        word_count = Counter()
        
        for text in tqdm(texts, desc="Building vocabulary"):
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split()
            word_count.update(words)
        
        # Reserve indices for special tokens
        self.word_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
        # Add most frequent words
        most_common = word_count.most_common(self.vocab_size - 4)
        for word, _ in most_common:
            self.word_to_idx[word] = len(self.word_to_idx)
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_built = True
        
        print(f"Vocabulary built with {len(self.word_to_idx)} words")
    
    def encode_text(self, text):
        """Encode text to token indices"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built yet. Call build_vocab() first.")
        
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        
        # Convert words to indices
        indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        
        # Truncate or pad
        if len(indices) > self.max_seq_len - 2:
            indices = indices[:self.max_seq_len - 2]
        
        # Add SOS and EOS tokens
        indices = [self.word_to_idx['<SOS>']] + indices + [self.word_to_idx['<EOS>']]
        
        # Pad to max length
        actual_length = len(indices)
        while len(indices) < self.max_seq_len:
            indices.append(self.word_to_idx['<PAD>'])
        
        return torch.tensor(indices), torch.tensor(actual_length)

class ImageDownloader:
    """Download and cache images from URLs"""
    def __init__(self, cache_dir="./image_cache", img_size=224):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.img_size = img_size
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def get_cache_path(self, url):
        """Get cache file path for URL"""
        url_hash = hash(url)
        return self.cache_dir / f"img_{url_hash}.jpg"
    
    def download_image(self, url, timeout=10):
        """Download image from URL with caching"""
        cache_path = self.get_cache_path(url)
        
        # Return cached image if exists
        if cache_path.exists():
            try:
                image = Image.open(cache_path).convert('RGB')
                return self.transform(image)
            except:
                cache_path.unlink()  # Remove corrupted cache
        
        # Download image
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Load and transform
            image = Image.open(cache_path).convert('RGB')
            return self.transform(image)
            
        except Exception as e:
            print(f"Failed to download image from {url}: {e}")
            # Return black image as fallback
            black_image = Image.new('RGB', (self.img_size, self.img_size), color='black')
            return self.transform(black_image)

class FakeNewsDataset(Dataset):
    """Generic fake news dataset class"""
    def __init__(self, data, text_preprocessor, image_downloader, encoder_type='RNN'):
        self.data = data.reset_index(drop=True)
        self.text_preprocessor = text_preprocessor
        self.image_downloader = image_downloader
        self.encoder_type = encoder_type
        
        if encoder_type == 'BERT':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get text
        text = str(row.get('text', ''))
        
        # Get image
        image_url = row.get('image_url', '')
        if pd.notna(image_url) and image_url:
            image = self.image_downloader.download_image(image_url)
        else:
            # Create black image if no URL
            black_image = Image.new('RGB', (224, 224), color='black')
            image = self.image_downloader.transform(black_image)
        
        # Get label
        label = int(row.get('label', 0))  # 0 for real, 1 for fake
        
        if self.encoder_type == 'RNN':
            # Encode text for RNN
            text_ids, length = self.text_preprocessor.encode_text(text)
            return text_ids, image, label, length
        else:
            # Encode text for BERT
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            text_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            return text_ids, attention_mask, image, label

class TwitterFakeNewsLoader:
    """Loader for Twitter fake news datasets"""
    
    @staticmethod
    def load_fakeddit(data_path, sample_size=None):
        """Load FakeDdit dataset (Reddit fake news with images)"""
        print("Loading FakeDdit dataset...")
        
        try:
            # Load the dataset
            df = pd.read_csv(data_path, sep='\t')
            
            # Map columns to standard format
            df['text'] = df['clean_title'].fillna('') + ' ' + df['clean_selftext'].fillna('')
            df['image_url'] = df['image_url']
            df['label'] = (df['2_way_label'] == 'fake').astype(int)
            
            # Filter only posts with images
            df = df[df['image_url'].notna() & (df['image_url'] != '')]
            
            # Sample if requested
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            print(f"Loaded {len(df)} samples with images")
            print(f"Label distribution: {df['label'].value_counts().to_dict()}")
            
            return df[['text', 'image_url', 'label']]
            
        except Exception as e:
            print(f"Error loading FakeDdit: {e}")
            return None
    
    @staticmethod
    def load_weibo(data_path, sample_size=None):
        """Load Weibo fake news dataset"""
        print("Loading Weibo dataset...")
        
        try:
            # This would load Weibo dataset - format may vary
            # Adjust based on actual dataset structure
            df = pd.read_json(data_path, lines=True)
            
            # Map to standard format (adjust column names as needed)
            df['text'] = df['text']
            df['image_url'] = df['image']
            df['label'] = df['label']
            
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            print(f"Loaded {len(df)} samples")
            print(f"Label distribution: {df['label'].value_counts().to_dict()}")
            
            return df[['text', 'image_url', 'label']]
            
        except Exception as e:
            print(f"Error loading Weibo: {e}")
            return None
    
    @staticmethod
    def create_sample_dataset(num_samples=1000):
        """Create a sample dataset for testing"""
        print("Creating sample dataset for testing...")
        
        fake_texts = [
            "BREAKING: Scientists discover cure for everything! Click here!",
            "You won't believe what this celebrity did! Shocking photos inside!",
            "Government hiding the truth about aliens! Secret documents leaked!",
            "This one weird trick will make you rich overnight!",
            "Doctors hate this simple method that cures all diseases!"
        ]
        
        real_texts = [
            "New study published in Nature shows promising results for cancer treatment",
            "Stock market shows modest gains amid economic uncertainty",
            "Climate scientists report new findings on global temperature trends",
            "Technology company announces quarterly earnings report",
            "Local community organizes charity drive for disaster relief"
        ]
        
        # Sample image URLs (placeholder images)
        image_urls = [
            "https://picsum.photos/300/300?random=1",
            "https://picsum.photos/300/300?random=2", 
            "https://picsum.photos/300/300?random=3",
            "https://picsum.photos/300/300?random=4",
            "https://picsum.photos/300/300?random=5"
        ]
        
        data = []
        for i in range(num_samples):
            if i % 2 == 0:  # Fake news
                text = np.random.choice(fake_texts)
                label = 1
            else:  # Real news
                text = np.random.choice(real_texts)
                label = 0
            
            image_url = np.random.choice(image_urls)
            data.append({
                'text': text,
                'image_url': image_url,
                'label': label
            })
        
        df = pd.DataFrame(data)
        print(f"Created {len(df)} samples")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df

class RealDatasetTrainer:
    """Trainer for real datasets with evaluation metrics"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.classification_loss = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    @staticmethod
    def collate_fn_rnn(batch):
        """Collate function for RNN"""
        text_ids, images, labels, lengths = zip(*batch)
        return (
            torch.stack(text_ids),
            torch.stack(images), 
            torch.tensor(labels),
            torch.stack(lengths)
        )
    
    @staticmethod
    def collate_fn_bert(batch):
        """Collate function for BERT"""
        text_ids, attention_masks, images, labels = zip(*batch)
        return (
            torch.stack(text_ids),
            torch.stack(attention_masks),
            torch.stack(images),
            torch.tensor(labels)
        )
    
    def train_epoch(self, dataloader, optimizer, alpha=0.7):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            try:
                # Handle different batch formats
                if len(batch) == 4 and torch.is_tensor(batch[3]) and batch[3].dim() == 1:  # RNN format
                    text_input, images, labels, lengths = batch
                    text_kwargs = {'lengths': lengths.to(self.device)}
                else:  # BERT format
                    text_input, attention_mask, images, labels = batch
                    text_kwargs = {'attention_mask': attention_mask.to(self.device)}
                
                text_input = text_input.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(text_input, images, **text_kwargs)
                
                # Classification loss
                cls_loss = self.classification_loss(logits, labels)
                
                # Contrastive loss
                cont_loss = self.model.compute_contrastive_loss(text_input, images, **text_kwargs)
                
                # Combined loss
                loss = (1 - alpha) * cls_loss + alpha * cont_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                try:
                    # Handle different batch formats
                    if len(batch) == 4 and torch.is_tensor(batch[3]) and batch[3].dim() == 1:  # RNN
                        text_input, images, labels, lengths = batch
                        text_kwargs = {'lengths': lengths.to(self.device)}
                    else:  # BERT
                        text_input, attention_mask, images, labels = batch
                        text_kwargs = {'attention_mask': attention_mask.to(self.device)}
                    
                    text_input = text_input.to(self.device)
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits = self.model(text_input, images, **text_kwargs)
                    loss = self.classification_loss(logits, labels)
                    
                    # Statistics
                    total_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in evaluation batch: {e}")
                    continue
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train_model(self, train_loader, val_loader, num_epochs=10, lr=0.001):
        """Complete training loop"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc, val_preds, val_labels = self.evaluate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        return self.history
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Accuracy')
        axes[1].plot(self.history['val_acc'], label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def detailed_evaluation(self, test_loader):
        """Detailed evaluation with metrics and confusion matrix"""
        test_loss, test_acc, test_preds, test_labels = self.evaluate(test_loader)
        
        # Calculate detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, test_preds, average='weighted'
        )
        
        print(f"\nDetailed Test Results:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

def main_training_pipeline():
    """Main training pipeline for real datasets"""
    print("=== Multimodal Fake News Detection - Real Dataset Training ===")
    
    # Configuration
    config = {
        'text_encoder_type': 'RNN',  # or 'BERT'
        'vocab_size': 10000,
        'max_seq_len': 128,
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'sample_size': 1000,  # Set to None to use full dataset
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    print(f"Configuration: {config}")
    
    # Step 1: Load dataset
    print("\n1. Loading Dataset...")
    
    # Option 1: Load real dataset (uncomment and provide path)
    # data = TwitterFakeNewsLoader.load_fakeddit('path/to/fakeddit.tsv', config['sample_size'])
    
    # Option 2: Use sample dataset for testing
    data = TwitterFakeNewsLoader.create_sample_dataset(config['sample_size'])
    
    if data is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Step 2: Split dataset
    print("\n2. Splitting Dataset...")
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42, stratify=data['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Step 3: Initialize preprocessors
    print("\n3. Initializing Preprocessors...")
    
    if config['text_encoder_type'] == 'RNN':
        text_preprocessor = TextPreprocessor(config['vocab_size'], config['max_seq_len'])
        text_preprocessor.build_vocab(train_data['text'].tolist())
    else:
        text_preprocessor = None
    
    image_downloader = ImageDownloader()
    
    # Step 4: Create datasets and dataloaders
    print("\n4. Creating Datasets and DataLoaders...")
    
    train_dataset = FakeNewsDataset(train_data, text_preprocessor, image_downloader, config['text_encoder_type'])
    val_dataset = FakeNewsDataset(val_data, text_preprocessor, image_downloader, config['text_encoder_type'])
    test_dataset = FakeNewsDataset(test_data, text_preprocessor, image_downloader, config['text_encoder_type'])
    
    # Create trainer and get appropriate collate function
    from main_model import create_model  # Import your model
    model = create_model(text_encoder_type=config['text_encoder_type'], vocab_size=config['vocab_size'])
    trainer = RealDatasetTrainer(model, config['device'])
    
    collate_fn = trainer.collate_fn_rnn if config['text_encoder_type'] == 'RNN' else trainer.collate_fn_bert
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # Step 5: Train model
    print("\n5. Training Model...")
    history = trainer.train_model(train_loader, val_loader, config['num_epochs'], config['learning_rate'])
    
    # Step 6: Plot training history
    print("\n6. Plotting Training History...")
    trainer.plot_training_history()
    
    # Step 7: Final evaluation
    print("\n7. Final Evaluation on Test Set...")
    results = trainer.detailed_evaluation(test_loader)
    
    # Step 8: Save results
    print("\n8. Saving Results...")
    
    # Save model
    torch.save(model.state_dict(), 'final_model.pth')
    
    # Save training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Save text preprocessor (for RNN)
    if config['text_encoder_type'] == 'RNN':
        with open('text_preprocessor.pkl', 'wb') as f:
            pickle.dump(text_preprocessor, f)
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in results.items()}
        json.dump(results_json, f, indent=2)
    
    print("\nTraining completed successfully!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Files saved:")
    print("- best_model.pth (best validation model)")
    print("- final_model.pth (final model)")
    print("- training_history.pkl")
    print("- text_preprocessor.pkl (if using RNN)")
    print("- evaluation_results.json")
    print("- training_history.png")
    print("- confusion_matrix.png")

if __name__ == "__main__":
    main_training_pipeline()