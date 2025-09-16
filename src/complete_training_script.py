#!/usr/bin/env python3
"""
Complete training script for multimodal fake news detection on real datasets
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import your existing modules
# Make sure these files are in the same directory or adjust imports
from main_model import create_model, FakeNewsTrainer  # Your original model code
from dataset_integration import *  # The dataset integration code
from dataset_downloaders import setup_real_datasets  # The dataset downloader code

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train multimodal fake news detection on real datasets')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='auto', 
                       choices=['auto', 'fakeddit', 'liar', 'gossipcop', 'weibo', 'combined'],
                       help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to dataset file (if not using auto-download)')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='Sample size for training (None for full dataset)')
    
    # Model arguments
    parser.add_argument('--text_encoder', type=str, default='RNN', 
                       choices=['RNN', 'BERT'],
                       help='Text encoder type')
    parser.add_argument('--rnn_type', type=str, default='LSTM',
                       choices=['LSTM', 'GRU'],
                       help='RNN type (if using RNN encoder)')
    parser.add_argument('--vocab_size', type=int, default=10000,
                       help='Vocabulary size for RNN encoder')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for RNN')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Weight for contrastive loss (1-alpha for classification loss)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()

def load_dataset(args):
    """Load and prepare dataset"""
    print(f"Loading dataset: {args.dataset}")
    
    if args.dataset == 'auto' or args.dataset == 'combined':
        # Try to load existing combined dataset
        combined_path = Path("./datasets/combined_fake_news_dataset.csv")
        if combined_path.exists():
            print("Loading existing combined dataset...")
            data = pd.read_csv(combined_path)
        else:
            print("Combined dataset not found. Setting up datasets...")
            data = setup_real_datasets()
            if data is None:
                raise ValueError("Failed to setup datasets")
    
    elif args.dataset_path:
        # Load from provided path
        print(f"Loading dataset from: {args.dataset_path}")
        data = pd.read_csv(args.dataset_path)
    
    else:
        # Load specific dataset
        processor = DatasetProcessor("./datasets")
        
        if args.dataset == 'fakeddit':
            data = processor.process_fakeddit('train', args.sample_size)
        elif args.dataset == 'liar':
            data = processor.process_liar('train', args.sample_size)
        elif args.dataset == 'gossipcop':
            data = processor.process_gossipcop(args.sample_size)
        elif args.dataset == 'weibo':
            data = processor.process_weibo(args.sample_size)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if data is None:
        raise ValueError("Failed to load dataset")
    
    # Sample if requested
    if args.sample_size and len(data) > args.sample_size:
        data = data.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    
    print(f"Dataset loaded: {len(data)} samples")
    print(f"Label distribution: {data['label'].value_counts().to_dict()}")
    
    return data

def create_data_loaders(data, args, text_preprocessor, image_downloader):
    """Create train/val/test data loaders"""
    
    # Split dataset
    train_data, temp_data = train_test_split(
        data, test_size=0.3, random_state=42, 
        stratify=data['label']
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42,
        stratify=temp_data['label']
    )
    
    print(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = FakeNewsDataset(train_data, text_preprocessor, image_downloader, args.text_encoder)
    val_dataset = FakeNewsDataset(val_data, text_preprocessor, image_downloader, args.text_encoder)
    test_dataset = FakeNewsDataset(test_data, text_preprocessor, image_downloader, args.text_encoder)
    
    # Use static methods for collate_fn
    collate_fn = RealDatasetTrainer.collate_fn_rnn if args.text_encoder == 'RNN' else RealDatasetTrainer.collate_fn_bert
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )
    
    return train_loader, val_loader, test_loader

def create_model_and_trainer(args):
    """Create model and trainer"""
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create model
    model_kwargs = {
        'text_encoder_type': args.text_encoder,
        'vocab_size': args.vocab_size,
    }
    
    if args.text_encoder == 'RNN':
        model_kwargs.update({
            'hidden_dim': args.hidden_dim,
            'rnn_type': args.rnn_type,
        })
    
    model = create_model(**model_kwargs)
    
    # Create trainer
    trainer = RealDatasetTrainer(model, device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, trainer

def save_results(args, model, trainer, results, text_preprocessor=None):
    """Save training results"""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    # Save training history
    with open(output_dir / 'training_history.pkl', 'wb') as f:
        pickle.dump(trainer.history, f)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save evaluation results
    results_json = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in results.items()}
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    # Save text preprocessor (if RNN)
    if text_preprocessor is not None:
        with open(output_dir / 'text_preprocessor.pkl', 'wb') as f:
            pickle.dump(text_preprocessor, f)
    
    print(f"Results saved to: {output_dir}")

def main():
    """Main training pipeline"""
    args = parse_arguments()
    
    print("=== Multimodal Fake News Detection Training ===")
    print(f"Configuration: {vars(args)}")
    
    try:
        # Step 1: Load dataset
        print("\n1. Loading Dataset...")
        data = load_dataset(args)
        
        # Step 2: Initialize preprocessors
        print("\n2. Initializing Preprocessors...")
        
        if args.text_encoder == 'RNN':
            text_preprocessor = TextPreprocessor(args.vocab_size, max_seq_len=128)
            text_preprocessor.build_vocab(data['text'].tolist())
        else:
            text_preprocessor = None
        
        image_downloader = ImageDownloader(cache_dir="./image_cache")
        
        # Step 3: Create data loaders
        print("\n3. Creating Data Loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data, args, text_preprocessor, image_downloader
        )
        
        # Step 4: Create model and trainer
        print("\n4. Creating Model and Trainer...")
        model, trainer = create_model_and_trainer(args)
        
        # Step 5: Resume from checkpoint if specified
        if args.resume:
            print(f"\n5. Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully")
        
        # Step 6: Train model
        print(f"\n6. Training Model for {args.num_epochs} epochs...")
        history = trainer.train_model(
            train_loader, val_loader, 
            num_epochs=args.num_epochs, 
            lr=args.learning_rate
        )
        
        # Step 7: Evaluate model
        print("\n7. Final Evaluation...")
        results = trainer.detailed_evaluation(test_loader)
        
        # Step 8: Save results
        print("\n8. Saving Results...")
        save_results(args, model, trainer, results, text_preprocessor)
        
        # Step 9: Print summary
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final Test Accuracy: {results['accuracy']:.4f}")
        print(f"Final Test F1-Score: {results['f1']:.4f}")
        print(f"Final Test Precision: {results['precision']:.4f}")
        print(f"Final Test Recall: {results['recall']:.4f}")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def quick_test():
    """Quick test with sample data"""
    print("=== Quick Test Mode ===")
    
    # Create sample arguments
    class Args:
        def __init__(self):
            self.dataset = 'auto'
            self.dataset_path = None
            self.sample_size = 1000
            self.text_encoder = 'RNN'
            self.rnn_type = 'LSTM'
            self.vocab_size = 5000
            self.hidden_dim = 256
            self.batch_size = 8
            self.num_epochs = 3
            self.learning_rate = 0.001
            self.alpha = 0.7
            self.device = 'auto'
            self.output_dir = './test_results'
            self.resume = None
    
    args = Args()
    
    # Use sample dataset instead of real data
    print("Creating sample dataset for quick test...")
    data = TwitterFakeNewsLoader.create_sample_dataset(args.sample_size)
    
    # Run training pipeline
    try:
        # Initialize preprocessors
        text_preprocessor = TextPreprocessor(args.vocab_size, max_seq_len=128)
        text_preprocessor.build_vocab(data['text'].tolist())
        image_downloader = ImageDownloader(cache_dir="./test_image_cache")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data, args, text_preprocessor, image_downloader
        )
        
        # Create model and trainer
        model, trainer = create_model_and_trainer(args)
        
        # Train model
        print("Starting quick training...")
        history = trainer.train_model(
            train_loader, val_loader, 
            num_epochs=args.num_epochs, 
            lr=args.learning_rate
        )
        
        # Evaluate
        results = trainer.detailed_evaluation(test_loader)
        
        # Save results
        save_results(args, model, trainer, results, text_preprocessor)
        
        print("Quick test completed successfully!")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Quick test mode
        quick_test()
    else:
        # Full training mode
        main()

# Additional utility functions for post-training analysis

def analyze_results(results_dir):
    """Analyze saved training results"""
    results_dir = Path(results_dir)
    
    # Load training history
    with open(results_dir / 'training_history.pkl', 'rb') as f:
        history = pickle.load(f)
    
    # Load evaluation results
    with open(results_dir / 'evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    # Plot training curves
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Confusion matrix
    cm = np.array(results['confusion_matrix'])
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Metrics summary
    metrics_text = f"""
    Test Results:
    Accuracy: {results['accuracy']:.4f}
    Precision: {results['precision']:.4f}
    Recall: {results['recall']:.4f}
    F1-Score: {results['f1']:.4f}
    
    Best Validation Accuracy: {max(history['val_acc']):.4f}
    Final Training Accuracy: {history['train_acc'][-1]:.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Results Summary')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analysis completed and saved to analysis.png")

def inference_example(model_path, preprocessor_path, config_path, text, image_url=None):
    """Example inference on new data"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    model_kwargs = {
        'text_encoder_type': config['text_encoder'],
        'vocab_size': config['vocab_size'],
    }
    
    if config['text_encoder'] == 'RNN':
        model_kwargs.update({
            'hidden_dim': config['hidden_dim'],
            'rnn_type': config['rnn_type'],
        })
    
    model = create_model(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load preprocessor
    if config['text_encoder'] == 'RNN':
        with open(preprocessor_path, 'rb') as f:
            text_preprocessor = pickle.load(f)
    else:
        from transformers import BertTokenizer
        text_preprocessor = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Initialize image downloader
    image_downloader = ImageDownloader()
    
    # Preprocess inputs
    if config['text_encoder'] == 'RNN':
        text_ids, length = text_preprocessor.encode_text(text)
        text_ids = text_ids.unsqueeze(0)  # Add batch dimension
        length = length.unsqueeze(0)
        text_kwargs = {'lengths': length}
    else:
        encoding = text_preprocessor(
            text, truncation=True, padding='max_length', 
            max_length=128, return_tensors='pt'
        )
        text_ids = encoding['input_ids']
        text_kwargs = {'attention_mask': encoding['attention_mask']}
    
    # Process image
    if image_url:
        image = image_downloader.download_image(image_url)
    else:
        from PIL import Image
        black_image = Image.new('RGB', (224, 224), color='black')
        image = image_downloader.transform(black_image)
    
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Inference
    with torch.no_grad():
        logits = model(text_ids, image, **text_kwargs)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    result = {
        'prediction': 'Fake' if prediction == 1 else 'Real',
        'confidence': confidence,
        'probabilities': {
            'Real': probabilities[0][0].item(),
            'Fake': probabilities[0][1].item()
        }
    }
    
    return result

# Usage examples and documentation
USAGE_EXAMPLES = """
Usage Examples:

1. Quick Test:
   python complete_training_script.py test

2. Train on Auto-downloaded Dataset:
   python complete_training_script.py --dataset auto --num_epochs 20

3. Train with BERT Encoder:
   python complete_training_script.py --text_encoder BERT --batch_size 8

4. Train on Specific Dataset:
   python complete_training_script.py --dataset fakeddit --sample_size 5000

5. Resume Training:
   python complete_training_script.py --resume ./results/best_model.pth

6. Full Training with Custom Parameters:
   python complete_training_script.py \
       --dataset combined \
       --text_encoder RNN \
       --rnn_type LSTM \
       --hidden_dim 512 \
       --batch_size 32 \
       --num_epochs 25 \
       --learning_rate 0.0005 \
       --output_dir ./custom_results

7. Analyze Results:
   python -c "from complete_training_script import analyze_results; analyze_results('./results')"

8. Inference Example:
   python -c "
   from complete_training_script import inference_example
   result = inference_example(
       './results/final_model.pth',
       './results/text_preprocessor.pkl',
       './results/config.json',
       'Breaking news: Scientists discover amazing cure!',
       'https://example.com/image.jpg'
   )
   print(result)
   "

Dataset Information:
- FakeDdit: Reddit posts with images, binary fake/real labels
- LIAR: Political statements, multi-class converted to binary
- GossipCop: Celebrity news with images
- Weibo: Chinese social media posts
- Combined: Mixture of all available datasets

Requirements:
- torch, torchvision
- transformers (for BERT)
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- PIL, requests
- tqdm

Setup:
1. Install requirements: pip install -r requirements.txt
2. Run quick test: python complete_training_script.py test
3. Download real datasets: python dataset_downloaders.py
4. Train on real data: python complete_training_script.py
"""

def print_usage():
    """Print usage information"""
    print(USAGE_EXAMPLES)