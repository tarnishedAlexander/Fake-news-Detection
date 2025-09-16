import os
import requests
import zipfile
import pandas as pd
import json
from pathlib import Path
import kaggle
from tqdm import tqdm
import gdown
import tarfile

class DatasetDownloader:
    """Download and setup various fake news datasets"""
    
    def __init__(self, data_dir="./datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_fakeddit(self):
        """Download FakeDdit dataset from GitHub"""
        print("Downloading FakeDdit dataset...")
        
        dataset_dir = self.data_dir / "fakeddit"
        dataset_dir.mkdir(exist_ok=True)
        
        # URLs for FakeDdit files
        urls = {
            "train": "https://github.com/entitize/Fakeddit/raw/master/data/train.tsv",
            "test": "https://github.com/entitize/Fakeddit/raw/master/data/test.tsv",
            "validate": "https://github.com/entitize/Fakeddit/raw/master/data/validate.tsv"
        }
        
        for split, url in urls.items():
            output_path = dataset_dir / f"{split}.tsv"
            if not output_path.exists():
                print(f"Downloading {split} split...")
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        for chunk in tqdm(response.iter_content(chunk_size=8192)):
                            f.write(chunk)
                    print(f"Downloaded {split}.tsv")
                except Exception as e:
                    print(f"Failed to download {split}: {e}")
            else:
                print(f"{split}.tsv already exists")
        
        print(f"FakeDdit dataset saved to: {dataset_dir}")
        return dataset_dir
    
    def download_gossipcop(self):
        """Download GossipCop dataset (requires manual setup)"""
        print("Setting up GossipCop dataset...")
        
        dataset_dir = self.data_dir / "gossipcop"
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"""
        To download GossipCop dataset:
        1. Visit: https://github.com/KaiDMML/FakeNewsNet
        2. Follow their instructions to download GossipCop data
        3. Extract to: {dataset_dir}
        
        The dataset contains:
        - gossipcop_fake.csv
        - gossipcop_real.csv
        - news content and social context data
        """)
        
        return dataset_dir
    
    def download_politifact(self):
        """Download PolitiFact dataset"""
        print("Setting up PolitiFact dataset...")
        
        dataset_dir = self.data_dir / "politifact"
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"""
        To download PolitiFact dataset:
        1. Visit: https://github.com/KaiDMML/FakeNewsNet
        2. Follow their instructions to download PolitiFact data
        3. Extract to: {dataset_dir}
        
        Alternative: Use the FakeNewsNet API
        """)
        
        return dataset_dir
    
    def download_weibo(self):
        """Download Weibo fake news dataset"""
        print("Setting up Weibo dataset...")
        
        dataset_dir = self.data_dir / "weibo"
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"""
        To download Weibo dataset:
        1. Visit: https://www.dropbox.com/s/7sr60306cbs91id/weibo.zip
        2. Download and extract to: {dataset_dir}
        
        Or use the direct download link in the code below
        """)
        
        # Try direct download
        try:
            url = "https://www.dropbox.com/s/7sr60306cbs91id/weibo.zip?dl=1"
            output_path = dataset_dir / "weibo.zip"
            
            if not output_path.exists():
                print("Attempting direct download...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192)):
                        f.write(chunk)
                
                # Extract zip
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                
                output_path.unlink()  # Remove zip file
                print("Weibo dataset downloaded and extracted")
        except Exception as e:
            print(f"Direct download failed: {e}")
        
        return dataset_dir
    
    def download_liar_dataset(self):
        """Download LIAR dataset"""
        print("Downloading LIAR dataset...")
        
        dataset_dir = self.data_dir / "liar"
        dataset_dir.mkdir(exist_ok=True)
        
        urls = {
            "train": "https://raw.githubusercontent.com/thiagocastroferreira/LIAR/master/dataset/train.tsv",
            "test": "https://raw.githubusercontent.com/thiagocastroferreira/LIAR/master/dataset/test.tsv",
            "valid": "https://raw.githubusercontent.com/thiagocastroferreira/LIAR/master/dataset/valid.tsv"
        }
        
        for split, url in urls.items():
            output_path = dataset_dir / f"{split}.tsv"
            if not output_path.exists():
                try:
                    print(f"Downloading {split} split...")
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {split}.tsv")
                except Exception as e:
                    print(f"Failed to download {split}: {e}")
        
        return dataset_dir
    
    def download_twitter_fakenews(self):
        """Download Twitter fake news dataset"""
        print("Setting up Twitter fake news dataset...")
        
        dataset_dir = self.data_dir / "twitter_fakenews"
        dataset_dir.mkdir(exist_ok=True)
        
        # Try to download from multiple sources
        sources = [
            {
                "name": "BuzzFeed Political News",
                "url": "https://raw.githubusercontent.com/BuzzFeedNews/2016-10-facebook-fact-check/master/data/facebook-fact-check.csv"
            }
        ]
        
        for source in sources:
            try:
                print(f"Downloading {source['name']}...")
                response = requests.get(source['url'])
                response.raise_for_status()
                
                filename = source['name'].lower().replace(' ', '_') + '.csv'
                with open(dataset_dir / filename, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {source['name']}: {e}")
        
        return dataset_dir
    
    def setup_kaggle_datasets(self):
        """Setup Kaggle datasets (requires Kaggle API)"""
        print("Setting up Kaggle datasets...")
        
        try:
            # Download Fake News Detection dataset
            kaggle_dir = self.data_dir / "kaggle"
            kaggle_dir.mkdir(exist_ok=True)
            
            datasets = [
                "clmentbisaillon/fake-and-real-news-dataset",
                "emineyetm/fake-news-detection-datasets",
                "therealsampat/fake-news-detection"
            ]
            
            for dataset in datasets:
                try:
                    print(f"Downloading {dataset}...")
                    kaggle.api.dataset_download_files(
                        dataset, 
                        path=kaggle_dir / dataset.split('/')[-1],
                        unzip=True
                    )
                    print(f"Downloaded {dataset}")
                except Exception as e:
                    print(f"Failed to download {dataset}: {e}")
            
        except Exception as e:
            print(f"Kaggle setup failed: {e}")
            print("Make sure you have:")
            print("1. Installed kaggle: pip install kaggle")
            print("2. Setup Kaggle API credentials (~/.kaggle/kaggle.json)")
        
        return self.data_dir / "kaggle"

class DatasetProcessor:
    """Process downloaded datasets into standard format"""
    
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
    
    def process_fakeddit(self, split='train', sample_size=None):
        """Process FakeDdit dataset"""
        print(f"Processing FakeDdit {split} split...")
        
        fakeddit_dir = self.dataset_dir / "fakeddit"
        file_path = fakeddit_dir / f"{split}.tsv"
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            
            # Filter for multimodal posts (has both text and image)
            df = df[df['hasImage'] == True]
            df = df.dropna(subset=['clean_title', 'image_url'])
            
            # Create standard format
            processed_df = pd.DataFrame({
                'text': df['clean_title'].fillna('') + ' ' + df['clean_selftext'].fillna(''),
                'image_url': df['image_url'],
                'label': (df['2_way_label'] == 'fake').astype(int),
                'source': 'fakeddit'
            })
            
            # Sample if requested
            if sample_size and len(processed_df) > sample_size:
                processed_df = processed_df.sample(n=sample_size, random_state=42)
            
            print(f"Processed {len(processed_df)} samples from FakeDdit {split}")
            print(f"Label distribution: {processed_df['label'].value_counts().to_dict()}")
            
            return processed_df
            
        except Exception as e:
            print(f"Error processing FakeDdit: {e}")
            return None
    
    def process_gossipcop(self, sample_size=None):
        """Process GossipCop dataset"""
        print("Processing GossipCop dataset...")
        
        gossipcop_dir = self.dataset_dir / "gossipcop"
        
        try:
            # Load fake and real news
            fake_df = pd.read_csv(gossipcop_dir / "gossipcop_fake.csv")
            real_df = pd.read_csv(gossipcop_dir / "gossipcop_real.csv")
            
            # Add labels
            fake_df['label'] = 1
            real_df['label'] = 0
            
            # Combine
            df = pd.concat([fake_df, real_df], ignore_index=True)
            
            # Create standard format
            processed_df = pd.DataFrame({
                'text': df['title'].fillna('') + ' ' + df['text'].fillna(''),
                'image_url': df.get('image_url', ''),
                'label': df['label'],
                'source': 'gossipcop'
            })
            
            # Filter for posts with images
            processed_df = processed_df[processed_df['image_url'].notna() & (processed_df['image_url'] != '')]
            
            if sample_size and len(processed_df) > sample_size:
                processed_df = processed_df.sample(n=sample_size, random_state=42)
            
            print(f"Processed {len(processed_df)} samples from GossipCop")
            return processed_df
            
        except Exception as e:
            print(f"Error processing GossipCop: {e}")
            return None
    
    def process_weibo(self, sample_size=None):
        """Process Weibo dataset"""
        print("Processing Weibo dataset...")
        
        weibo_dir = self.dataset_dir / "weibo"
        
        try:
            # Load Weibo data (format may vary)
            json_files = list(weibo_dir.glob("*.json"))
            
            if not json_files:
                print("No JSON files found in Weibo directory")
                return None
            
            all_data = []
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            
            df = pd.DataFrame(all_data)
            
            # Create standard format (adjust column names as needed)
            processed_df = pd.DataFrame({
                'text': df['text'],
                'image_url': df.get('image_url', ''),
                'label': df['label'],
                'source': 'weibo'
            })
            
            if sample_size and len(processed_df) > sample_size:
                processed_df = processed_df.sample(n=sample_size, random_state=42)
            
            print(f"Processed {len(processed_df)} samples from Weibo")
            return processed_df
            
        except Exception as e:
            print(f"Error processing Weibo: {e}")
            return None
    
    def process_liar(self, split='train', sample_size=None):
        """Process LIAR dataset"""
        print(f"Processing LIAR {split} split...")
        
        liar_dir = self.dataset_dir / "liar"
        file_path = liar_dir / f"{split}.tsv"
        
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return None
        
        try:
            # LIAR dataset columns
            columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 
                      'state', 'party', 'barely_true_counts', 'false_counts',
                      'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
            
            df = pd.read_csv(file_path, sep='\t', names=columns)
            
            # Convert labels to binary (false/pants-fire vs others)
            fake_labels = ['false', 'pants-fire', 'barely-true']
            df['binary_label'] = df['label'].apply(lambda x: 1 if x in fake_labels else 0)
            
            # Create standard format
            processed_df = pd.DataFrame({
                'text': df['statement'],
                'image_url': '',  # LIAR doesn't have images
                'label': df['binary_label'],
                'source': 'liar'
            })
            
            if sample_size and len(processed_df) > sample_size:
                processed_df = processed_df.sample(n=sample_size, random_state=42)
            
            print(f"Processed {len(processed_df)} samples from LIAR {split}")
            return processed_df
            
        except Exception as e:
            print(f"Error processing LIAR: {e}")
            return None
    
    def combine_datasets(self, datasets, output_path=None):
        """Combine multiple processed datasets"""
        print("Combining datasets...")
        
        valid_datasets = [df for df in datasets if df is not None]
        
        if not valid_datasets:
            print("No valid datasets to combine")
            return None
        
        combined_df = pd.concat(valid_datasets, ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        print(f"Combined dataset: {len(combined_df)} samples")
        print(f"Label distribution: {combined_df['label'].value_counts().to_dict()}")
        print(f"Source distribution: {combined_df['source'].value_counts().to_dict()}")
        
        if output_path:
            combined_df.to_csv(output_path, index=False)
            print(f"Saved combined dataset to: {output_path}")
        
        return combined_df

def setup_real_datasets():
    """Complete setup pipeline for real datasets"""
    print("=== Setting up Real Fake News Datasets ===")
    
    # Initialize downloaders
    downloader = DatasetDownloader("./datasets")
    processor = DatasetProcessor("./datasets")
    
    # Download datasets
    print("\n1. Downloading datasets...")
    
    datasets_info = []
    
    # Download FakeDdit
    try:
        fakeddit_dir = downloader.download_fakeddit()
        datasets_info.append(("fakeddit", fakeddit_dir))
    except Exception as e:
        print(f"FakeDdit download failed: {e}")
    
    # Download LIAR
    try:
        liar_dir = downloader.download_liar_dataset()
        datasets_info.append(("liar", liar_dir))
    except Exception as e:
        print(f"LIAR download failed: {e}")
    
    # Download Twitter datasets
    try:
        twitter_dir = downloader.download_twitter_fakenews()
        datasets_info.append(("twitter", twitter_dir))
    except Exception as e:
        print(f"Twitter download failed: {e}")
    
    # Setup other datasets (manual)
    downloader.download_gossipcop()
    downloader.download_weibo()
    
    # Process datasets
    print("\n2. Processing datasets...")
    
    processed_datasets = []
    
    # Process FakeDdit
    fakeddit_train = processor.process_fakeddit('train', sample_size=5000)
    if fakeddit_train is not None:
        processed_datasets.append(fakeddit_train)
    
    # Process LIAR
    liar_train = processor.process_liar('train', sample_size=5000)
    if liar_train is not None:
        processed_datasets.append(liar_train)
    
    # Process GossipCop (if available)
    try:
        gossipcop_data = processor.process_gossipcop(sample_size=2000)
        if gossipcop_data is not None:
            processed_datasets.append(gossipcop_data)
    except:
        print("GossipCop processing skipped (data not available)")
    
    # Combine all datasets
    print("\n3. Combining datasets...")
    combined_dataset = processor.combine_datasets(
        processed_datasets, 
        output_path="./datasets/combined_fake_news_dataset.csv"
    )
    
    # Create usage instructions
    if combined_dataset is not None:
        print("\n4. Dataset ready for training!")
        print("Usage:")
        print("```python")
        print("import pandas as pd")
        print("data = pd.read_csv('./datasets/combined_fake_news_dataset.csv')")
        print("print(f'Dataset size: {len(data)}')")
        print("print(f'Label distribution: {data[\"label\"].value_counts()}')")
        print("```")
        
        return combined_dataset
    else:
        print("Failed to create combined dataset")
        return None

if __name__ == "__main__":
    # Setup real datasets
    dataset = setup_real_datasets()
    
    if dataset is not None:
        print("\nDataset setup completed successfully!")
        print(f"Combined dataset saved with {len(dataset)} samples")
    else:
        print("\nDataset setup failed. Please check the instructions above for manual downloads.")