import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from transformers import BertModel, BertConfig

class PatchEmbed(nn.Module):
    """Image to Patch Embedding for ViT"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention for ViT"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    """MLP block for ViT"""
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block for ViT"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer for image encoding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return class token representation
        return x[:, 0]

class RNNTextEncoder(nn.Module):
    """RNN-based text encoder"""
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512, num_layers=2, rnn_type='LSTM'):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        
        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, num_layers=num_layers,
                bidirectional=True, dropout=0.3 if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, num_layers=num_layers,
                bidirectional=True, dropout=0.3 if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Project to final embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, lengths=None):
        # x: (batch_size, seq_len)
        embeddings = self.word_embedding(x)  # (batch_size, seq_len, embed_dim)
        
        if lengths is not None:
            # Pack padded sequences
            embeddings = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # RNN forward pass
        rnn_out, _ = self.rnn(embeddings)
        
        if lengths is not None:
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        
        # Mean pooling with masking
        if lengths is not None:
            mask = torch.arange(rnn_out.size(1)).expand(len(lengths), rnn_out.size(1)).to(rnn_out.device)
            mask = mask < lengths.unsqueeze(1)
            rnn_out = rnn_out * mask.unsqueeze(-1).float()
            pooled = rnn_out.sum(1) / lengths.unsqueeze(1).float()
        else:
            pooled = rnn_out.mean(1)
        
        # Project to final embedding
        final_embedding = self.projection(pooled)
        return final_embedding

class BERTTextEncoder(nn.Module):
    """BERT-based text encoder"""
    def __init__(self, model_name='bert-base-uncased', embed_dim=256, freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Project BERT output to desired embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, input_ids, attention_mask=None, lengths=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0]  # (batch_size, bert_hidden_size)
        
        # Project to final embedding
        final_embedding = self.projection(cls_output)
        return final_embedding

class FlexibleTextEncoder(nn.Module):
    """Flexible text encoder that can switch between RNN and BERT"""
    def __init__(self, encoder_type='RNN', vocab_size=10000, embed_dim=256, **kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'RNN':
            self.encoder = RNNTextEncoder(vocab_size, embed_dim, **kwargs)
        elif encoder_type == 'BERT':
            self.encoder = BERTTextEncoder(embed_dim=embed_dim, **kwargs)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    def forward(self, x, **kwargs):
        return self.encoder(x, **kwargs)

class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for multimodal learning"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, text_embeddings, image_embeddings):
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, dim=1)
        image_embeddings = F.normalize(image_embeddings, dim=1)
        
        # Calculate similarity matrix
        similarity = torch.matmul(text_embeddings, image_embeddings.T) / self.temperature
        
        # Create labels (diagonal elements are positive pairs)
        batch_size = text_embeddings.size(0)
        labels = torch.arange(batch_size).to(similarity.device)
        
        # Calculate InfoNCE loss for text->image and image->text
        loss_t2i = F.cross_entropy(similarity, labels)
        loss_i2t = F.cross_entropy(similarity.T, labels)
        
        return (loss_t2i + loss_i2t) / 2

class MultimodalFakeNewsDetector(nn.Module):
    """Complete multimodal fake news detection model with ViT + flexible text encoder"""
    def __init__(self, 
                 # Text encoder parameters
                 text_encoder_type='RNN',  # 'RNN' or 'BERT'
                 vocab_size=10000,
                 text_embed_dim=256,
                 # ViT parameters
                 img_size=224,
                 patch_size=16,
                 vit_embed_dim=768,
                 vit_depth=12,
                 vit_num_heads=12,
                 # Classification parameters
                 num_classes=2,
                 fusion_dim=256,
                 **text_encoder_kwargs):
        super().__init__()
        
        # Text encoder (flexible RNN or BERT)
        self.text_encoder = FlexibleTextEncoder(
            encoder_type=text_encoder_type,
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            **text_encoder_kwargs
        )
        
        # Vision transformer
        self.image_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads
        )
        
        # Project ViT output to same dimension as text
        self.image_projection = nn.Sequential(
            nn.Linear(vit_embed_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Project text output to fusion dimension
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss()
        
    def encode_text(self, text_input, **kwargs):
        text_features = self.text_encoder(text_input, **kwargs)
        return self.text_projection(text_features)
    
    def encode_image(self, images):
        image_features = self.image_encoder(images)
        return self.image_projection(image_features)
    
    def forward(self, text_input, images, return_embeddings=False, **text_kwargs):
        # Encode modalities
        text_embeddings = self.encode_text(text_input, **text_kwargs)
        image_embeddings = self.encode_image(images)
        
        if return_embeddings:
            return text_embeddings, image_embeddings
        
        # Concatenate embeddings for classification
        combined = torch.cat([text_embeddings, image_embeddings], dim=1)
        logits = self.classifier(combined)
        
        return logits
    
    def compute_contrastive_loss(self, text_input, images, **text_kwargs):
        """Compute contrastive loss between text and image embeddings"""
        text_embeddings, image_embeddings = self.forward(
            text_input, images, return_embeddings=True, **text_kwargs
        )
        return self.contrastive_loss(text_embeddings, image_embeddings)

# Training utilities
class FakeNewsTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.classification_loss = nn.CrossEntropyLoss()
        
    def train_step(self, batch, optimizer, alpha=0.7):
        """Training step with combined contrastive and classification loss"""
        # Handle different batch formats for RNN vs BERT
        if len(batch) == 4:  # RNN format: text_ids, images, labels, lengths
            text_input, images, labels, lengths = batch
            text_kwargs = {'lengths': lengths.to(self.device) if lengths is not None else None}
        else:  # BERT format: text_ids, attention_mask, images, labels
            text_input, attention_mask, images, labels = batch
            text_kwargs = {'attention_mask': attention_mask.to(self.device)}
            
        text_input = text_input.to(self.device)
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(text_input, images, **text_kwargs)
        
        # Classification loss
        cls_loss = self.classification_loss(logits, labels)
        
        # Contrastive loss
        cont_loss = self.model.compute_contrastive_loss(text_input, images, **text_kwargs)
        
        # Combined loss
        total_loss = (1 - alpha) * cls_loss + alpha * cont_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()
        
        return {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'cont_loss': cont_loss.item(),
            'accuracy': accuracy.item()
        }

# Factory function to create models
def create_model(text_encoder_type='RNN', **kwargs):
    """Factory function to create models with different text encoders"""
    if text_encoder_type == 'RNN':
        return MultimodalFakeNewsDetector(
            text_encoder_type='RNN',
            hidden_dim=kwargs.get('hidden_dim', 512),
            num_layers=kwargs.get('num_layers', 2),
            rnn_type=kwargs.get('rnn_type', 'LSTM'),
            **{k: v for k, v in kwargs.items() if k not in ['hidden_dim', 'num_layers', 'rnn_type']}
        )
    elif text_encoder_type == 'BERT':
        return MultimodalFakeNewsDetector(
            text_encoder_type='BERT',
            model_name=kwargs.get('model_name', 'bert-base-uncased'),
            freeze_bert=kwargs.get('freeze_bert', False),
            **{k: v for k, v in kwargs.items() if k not in ['model_name', 'freeze_bert']}
        )
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

if __name__ == "__main__":
    # Example 1: ViT + RNN
    print("Creating ViT + RNN model...")
    model_rnn = create_model(
        text_encoder_type='RNN',
        vocab_size=10000,
        hidden_dim=512,
        rnn_type='LSTM'
    )
    print(f"RNN Model parameters: {sum(p.numel() for p in model_rnn.parameters()):,}")
    
    # Example 2: ViT + BERT
    print("\nCreating ViT + BERT model...")
    try:
        model_bert = create_model(
            text_encoder_type='BERT',
            freeze_bert=False
        )
        print(f"BERT Model parameters: {sum(p.numel() for p in model_bert.parameters()):,}")
    except Exception as e:
        print(f"BERT model creation failed: {e}")
        print("This is expected if transformers library is not installed")
    
    # Test with sample data
    batch_size = 2
    text_ids = torch.randint(1, 1000, (batch_size, 50))
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size,))
    lengths = torch.tensor([45, 50])
    
    # Test RNN model
    print("\nTesting RNN model...")
    with torch.no_grad():
        logits_rnn = model_rnn(text_ids, images, lengths=lengths)
        print(f"RNN output shape: {logits_rnn.shape}")
    
    print("\nModels ready for training!")