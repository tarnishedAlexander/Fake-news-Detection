import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from bert import BERTFakeNewsClassifier
from rnn import RNNFakeNewsClassifier, build_vocabulary, preprocess_text

def load_datasets():
    """Load and combine both true and fake news datasets"""
    print("Loading datasets...")
    
    dataset_files = {
        'true': '../dataset/dataset/datasets/True.csv',
        'false': '../dataset/dataset/datasets/Fake.csv'
    }
    
    dataframes = []
    
    for label_name, file_path in dataset_files.items():
        try:
            print(f"Loading {file_path}...")
            df = pd.read_csv(file_path)
            df['binary_label'] = 0 if label_name == 'true' else 1
            df['source_file'] = label_name
            print(f"   Loaded: {len(df)} {label_name} news samples")
            dataframes.append(df)
        except Exception as e:
            rnn_classifier = RNNFakeNewsClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.4,
        rnn_type='LSTM',
        bidirectional=True
    )
    
    rnn_classifier.prepare_data(X_train, X_val, X_test, y_train, y_val, y_test, vocab_to_idx)
    
    rnn_classifier.train(num_epochs=10, learning_rate=0.001, batch_size=32)
    
    test_loss, test_acc, predictions, true_labels, report, cm = rnn_classifier.test_model()
    
    print(f"RNN Test Accuracy: {test_acc:.4f}")
    
    return rnn_classifier, test_acc

def plot_model_comparison(bert_classifier, rnn_classifier):
    """Plot comparison of both models' training progress"""
    print("\nGenerating comparison plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BERT vs RNN Comparison', fontsize=16)
    
    # Plot 1: Training Loss Comparison
    axes[0,0].plot(bert_classifier.train_losses, label='BERT', marker='o', linewidth=2)
    axes[0,0].plot(rnn_classifier.train_losses, label='RNN', marker='s', linewidth=2)
    axes[0,0].set_title('Training Loss Comparison')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss Comparison
    axes[0,1].plot(bert_classifier.val_losses, label='BERT', marker='o', linewidth=2)
    axes[0,1].plot(rnn_classifier.val_losses, label='RNN', marker='s', linewidth=2)
    axes[0,1].set_title('Validation Loss Comparison')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy Comparison
    axes[1,0].plot(bert_classifier.val_accuracies, label='BERT', marker='o', linewidth=2)
    axes[1,0].plot(rnn_classifier.val_accuracies, label='RNN', marker='s', linewidth=2)
    axes[1,0].set_title('Validation Accuracy Comparison')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Training Accuracy Comparison (for RNN only, BERT doesn't track this)
    if hasattr(rnn_classifier, 'train_accuracies') and rnn_classifier.train_accuracies:
        axes[1,1].plot(rnn_classifier.train_accuracies, label='RNN Training', marker='s', linewidth=2)
        axes[1,1].plot(rnn_classifier.val_accuracies, label='RNN Validation', marker='o', linewidth=2)
        axes[1,1].set_title('RNN Training vs Validation Accuracy')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Model comparison plot saved as 'model_comparison.png'")

def plot_final_accuracy_comparison(bert_acc, rnn_acc):
    """Plot final test accuracy comparison"""
    models = ['BERT', 'RNN']
    accuracies = [bert_acc, rnn_acc]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    plt.title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('final_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Final accuracy comparison saved as 'final_accuracy_comparison.png'")

def test_predictions(bert_classifier, rnn_classifier, X_test, y_test):
    """Test both models on sample texts and show predictions"""
    print("\nTesting sample predictions...")
    
    sample_texts = X_test[:5]
    sample_labels = y_test[:5]
    
    bert_predictions, bert_probs = bert_classifier.predict(sample_texts)
    rnn_predictions, rnn_probs = rnn_classifier.predict(sample_texts)
    
    print(f"\n{'='*60}")
    print("Sample Predictions Comparison")
    print(f"{'='*60}")
    
    for i, (text, true_label) in enumerate(zip(sample_texts, sample_labels)):
        bert_pred = bert_predictions[i]
        rnn_pred = rnn_predictions[i]
        
        bert_conf = max(bert_probs[i]) * 100
        rnn_conf = max(rnn_probs[i]) * 100
        
        true_name = "FAKE" if true_label == 1 else "REAL"
        bert_name = "FAKE" if bert_pred == 1 else "REAL"
        rnn_name = "FAKE" if rnn_pred == 1 else "REAL"
        
        bert_correct = "✓" if bert_pred == true_label else "✗"
        rnn_correct = "✓" if rnn_pred == true_label else "✗"
        
        print(f"\nSample {i+1}:")
        print(f"Text: {text[:100]}...")
        print(f"True Label: {true_name}")
        print(f"BERT: {bert_name} ({bert_conf:.1f}%) {bert_correct}")
        print(f"RNN:  {rnn_name} ({rnn_conf:.1f}%) {rnn_correct}")

def main():
    """Main training and comparison function"""
    print("="*60)
    print("BERT vs RNN Fake News Classification Comparison")
    print("="*60)
    
    # Load datasets
    try:
        df, text_column = load_datasets()
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure the following files exist:")
        print("- ../dataset/dataset/datasets/True.csv")
        print("- ../dataset/dataset/datasets/Fake.csv")
        return
    
    # Create balanced dataset
    texts, labels = create_balanced_dataset(df, text_column, max_samples_per_class=5000)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)
    
    # Train both models
    bert_classifier, bert_accuracy = train_bert_model(X_train, X_val, X_test, y_train, y_val, y_test)
    rnn_classifier, rnn_accuracy = train_rnn_model(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Generate comparison plots
    plot_model_comparison(bert_classifier, rnn_classifier)
    plot_final_accuracy_comparison(bert_accuracy, rnn_accuracy)
    
    # Test sample predictions
    test_predictions(bert_classifier, rnn_classifier, X_test, y_test)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("Final Results Summary")
    print(f"{'='*60}")
    print(f"BERT Test Accuracy: {bert_accuracy:.4f} ({bert_accuracy*100:.2f}%)")
    print(f"RNN Test Accuracy:  {rnn_accuracy:.4f} ({rnn_accuracy*100:.2f}%)")
    
    if bert_accuracy > rnn_accuracy:
        diff = (bert_accuracy - rnn_accuracy) * 100
        print(f"BERT outperforms RNN by {diff:.2f} percentage points")
    elif rnn_accuracy > bert_accuracy:
        diff = (rnn_accuracy - bert_accuracy) * 100
        print(f"RNN outperforms BERT by {diff:.2f} percentage points")
    else:
        print("Both models achieved the same accuracy")
    
    print("\nFiles generated:")
    print("- model_comparison.png: Training progress comparison")
    print("- final_accuracy_comparison.png: Test accuracy comparison")
    print("- best_rnn_model.pth: Best RNN model weights")
    print("- bert_fake_news_model/: BERT model directory")

if __name__ == "__main__":
    main()aise FileNotFoundError(f"Error loading {file_path}: {str(e)}")
    
    if len(dataframes) == 0:
        raise FileNotFoundError("No dataset files found")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Find text column
    text_column = None
    for col in ['text', 'title']:
        if col in combined_df.columns:
            text_column = col
            break
    
    if text_column is None:
        text_columns = combined_df.select_dtypes(include=['object']).columns
        if len(text_columns) > 0:
            text_column = text_columns[0]
    
    if text_column is None:
        raise ValueError("No text column found in dataset")
    
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Text column: '{text_column}'")
    
    distribution = combined_df['binary_label'].value_counts()
    for label, count in distribution.items():
        label_name = "Fake News" if label == 1 else "Real News"
        percentage = (count / len(combined_df)) * 100
        print(f"  {label} ({label_name}): {count} ({percentage:.1f}%)")
    
    return combined_df, text_column

def create_balanced_dataset(df, text_column, max_samples_per_class=None):
    """Create a balanced dataset"""
    print("\nBalancing dataset...")
    
    class_counts = df['binary_label'].value_counts()
    min_samples = class_counts.min()
    
    if max_samples_per_class:
        min_samples = min(max_samples_per_class, min_samples)
    
    print(f"Using {min_samples} samples per class...")
    
    balanced_dfs = []
    for label in df['binary_label'].unique():
        class_df = df[df['binary_label'] == label].sample(n=min_samples, random_state=42)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    texts = balanced_df[text_column].fillna('').tolist()
    labels = balanced_df['binary_label'].tolist()
    
    print(f"Balanced dataset created: {len(texts)} total samples")
    balanced_counts = balanced_df['binary_label'].value_counts()
    for label, count in balanced_counts.items():
        label_name = "Fake News" if label == 1 else "Real News"
        print(f"  {label} ({label_name}): {count}")
    
    return texts, labels

def split_data(texts, labels, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets"""
    print(f"\nSplitting data...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=test_size+val_size, random_state=42, stratify=labels
    )
    
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_bert_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train BERT model"""
    print("\n" + "="*50)
    print("Training BERT Model")
    print("="*50)
    
    bert_classifier = BERTFakeNewsClassifier(model_name='distilbert-base-uncased')
    bert_classifier.prepare_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    bert_classifier.train(epochs=3, batch_size=16, learning_rate=2e-5)
    
    test_loss, test_acc, predictions, true_labels, report, cm = bert_classifier.test_model()
    
    print(f"BERT Test Accuracy: {test_acc:.4f}")
    
    return bert_classifier, test_acc

def train_rnn_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train RNN model"""
    print("\n" + "="*50)
    print("Training RNN Model")
    print("="*50)
    
    # Build vocabulary from training data
    processed_texts = [preprocess_text(text) for text in X_train]
    vocab_to_idx, vocab = build_vocabulary(processed_texts, min_freq=3, max_vocab_size=15000)
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")