"""
Multi-Modal Unified Intelligence System
Implements vision, audio, code, and math understanding in a unified framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from einops import rearrange, repeat
import logging

logger = logging.getLogger(__name__)


class Modality(Enum):
    """Supported modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"
    MATH = "math"
    VIDEO = "video"


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal system"""
    # Text
    text_vocab_size: int = 100352
    max_text_length: int = 8192
    
    # Vision
    image_size: int = 224
    patch_size: int = 14
    num_image_tokens: int = 256
    vision_layers: int = 24
    
    # Audio
    audio_sample_rate: int = 16000
    n_mels: int = 80
    audio_frame_size: int = 400
    audio_hop_size: int = 160
    max_audio_length: int = 30  # seconds
    
    # Code
    code_vocab_size: int = 50000
    max_code_length: int = 4096
    syntax_aware: bool = True
    
    # Math
    math_vocab_size: int = 10000
    symbolic_math: bool = True
    
    # Shared
    hidden_dim: int = 4096
    num_heads: int = 32
    dropout: float = 0.1
    
    # Cross-modal
    cross_attention_layers: List[int] = None
    modal_dropout: float = 0.2
    fusion_type: str = "adaptive"  # adaptive, concatenate, cross_attention


class ModalityEmbedding(nn.Module):
    """Modality-specific embeddings"""
    
    def __init__(self, num_modalities: int, hidden_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_modalities, hidden_dim)
        
    def forward(self, modality_id: int, shape: Tuple[int, ...]) -> torch.Tensor:
        """Get modality embedding expanded to shape"""
        embedding = self.embeddings(torch.tensor(modality_id))
        return embedding.expand(*shape, -1)


class VisionEncoder(nn.Module):
    """Vision Transformer encoder for images"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, config.hidden_dim) * 0.02
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
        # Vision transformer layers
        self.layers = nn.ModuleList([
            VisionTransformerLayer(config)
            for _ in range(config.vision_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Projection to main model dimension
        self.projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to token representations"""
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, hidden_dim, H', W']
        x = rearrange(x, 'b d h w -> b (h w) d')
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Vision transformer
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Project to main model dimension
        x = self.projection(x)
        
        return x


class VisionTransformerLayer(nn.Module):
    """Single vision transformer layer"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = residual + x
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class AudioEncoder(nn.Module):
    """Audio encoder using mel-spectrograms and CNN/RNN"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        
        self.config = config
        
        # Mel-spectrogram parameters
        self.n_mels = config.n_mels
        self.frame_size = config.audio_frame_size
        self.hop_size = config.audio_hop_size
        
        # CNN for local features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        
        # Calculate CNN output size
        conv_out_size = self._calculate_conv_output_size()
        
        # RNN for temporal modeling
        self.rnn = nn.LSTM(
            conv_out_size,
            config.hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Projection to main model dimension
        self.projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size of CNN layers"""
        # Simplified calculation - would need actual computation
        return 512 * (self.n_mels // 4)
    
    def compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert raw audio to mel-spectrogram"""
        # Placeholder - would use torchaudio or librosa
        batch_size = audio.shape[0]
        n_frames = audio.shape[1] // self.hop_size
        mel_spec = torch.randn(batch_size, 1, self.n_mels, n_frames)
        return mel_spec
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to token representations"""
        # Convert to mel-spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)
        
        # CNN encoding
        conv_out = self.conv_layers(mel_spec)
        
        # Reshape for RNN
        batch_size, channels, freq, time = conv_out.shape
        conv_out = rearrange(conv_out, 'b c f t -> b t (c f)')
        
        # RNN encoding
        rnn_out, _ = self.rnn(conv_out)
        
        # Project to model dimension
        output = self.projection(rnn_out)
        
        return output


class CodeEncoder(nn.Module):
    """Code-aware encoder with syntax understanding"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        
        self.config = config
        
        # Code tokenizer (would use specialized tokenizer)
        self.code_embeddings = nn.Embedding(config.code_vocab_size, config.hidden_dim)
        
        # Syntax-aware components
        if config.syntax_aware:
            # AST node type embeddings
            self.ast_type_embeddings = nn.Embedding(100, config.hidden_dim // 4)
            
            # Indentation level embeddings
            self.indent_embeddings = nn.Embedding(20, config.hidden_dim // 4)
            
            # Combine embeddings
            self.combine_proj = nn.Linear(
                config.hidden_dim + config.hidden_dim // 2,
                config.hidden_dim
            )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_code_length, config.hidden_dim)
        
        # Code understanding layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim * 4,
                config.dropout,
                batch_first=True
            )
            for _ in range(6)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def extract_syntax_features(self, code_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract syntax features from code"""
        # Placeholder - would use actual AST parser
        batch_size, seq_len = code_tokens.shape
        ast_types = torch.randint(0, 100, (batch_size, seq_len))
        indent_levels = torch.randint(0, 20, (batch_size, seq_len))
        return ast_types, indent_levels
    
    def forward(self, code_tokens: torch.Tensor) -> torch.Tensor:
        """Encode code to token representations"""
        batch_size, seq_len = code_tokens.shape
        
        # Token embeddings
        token_embeds = self.code_embeddings(code_tokens)
        
        # Add syntax awareness
        if self.config.syntax_aware:
            ast_types, indent_levels = self.extract_syntax_features(code_tokens)
            ast_embeds = self.ast_type_embeddings(ast_types)
            indent_embeds = self.indent_embeddings(indent_levels)
            
            # Combine all embeddings
            combined = torch.cat([token_embeds, ast_embeds, indent_embeds], dim=-1)
            embeddings = self.combine_proj(combined)
        else:
            embeddings = token_embeds
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=code_tokens.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = embeddings + self.position_embeddings(positions)
        
        # Transform through layers
        x = embeddings
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x


class MathEncoder(nn.Module):
    """Mathematical expression encoder with symbolic understanding"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        
        self.config = config
        
        # Math token embeddings
        self.math_embeddings = nn.Embedding(config.math_vocab_size, config.hidden_dim)
        
        # Symbolic components
        if config.symbolic_math:
            # Operator type embeddings
            self.operator_embeddings = nn.Embedding(50, config.hidden_dim // 4)
            
            # Precedence level embeddings
            self.precedence_embeddings = nn.Embedding(10, config.hidden_dim // 4)
            
            # Combine projection
            self.combine_proj = nn.Linear(
                config.hidden_dim + config.hidden_dim // 2,
                config.hidden_dim
            )
        
        # Math-specific transformer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim * 4,
                config.dropout,
                batch_first=True
            )
            for _ in range(4)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def extract_math_features(self, math_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract mathematical structure features"""
        # Placeholder - would use actual math parser
        batch_size, seq_len = math_tokens.shape
        operators = torch.randint(0, 50, (batch_size, seq_len))
        precedence = torch.randint(0, 10, (batch_size, seq_len))
        return operators, precedence
    
    def forward(self, math_tokens: torch.Tensor) -> torch.Tensor:
        """Encode mathematical expressions"""
        batch_size, seq_len = math_tokens.shape
        
        # Token embeddings
        token_embeds = self.math_embeddings(math_tokens)
        
        # Add symbolic understanding
        if self.config.symbolic_math:
            operators, precedence = self.extract_math_features(math_tokens)
            op_embeds = self.operator_embeddings(operators)
            prec_embeds = self.precedence_embeddings(precedence)
            
            combined = torch.cat([token_embeds, op_embeds, prec_embeds], dim=-1)
            embeddings = self.combine_proj(combined)
        else:
            embeddings = token_embeds
        
        # Transform
        x = embeddings
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x


class CrossModalAttention(nn.Module):
    """Cross-attention between different modalities"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Cross-attention layers
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Gating mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
    def forward(
        self,
        query_modality: torch.Tensor,
        key_value_modality: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cross-modal attention"""
        residual = query_modality
        
        # Normalize
        query = self.norm1(query_modality)
        key_value = self.norm1(key_value_modality)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(query, key_value, key_value, attn_mask=attention_mask)
        
        # Adaptive gating
        gate_input = torch.cat([query_modality.mean(dim=1), attn_out.mean(dim=1)], dim=-1)
        gate_input = gate_input.unsqueeze(1).expand_as(attn_out)
        gate = self.gate(gate_input)
        
        # Gated residual connection
        x = residual + gate * attn_out
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class MultiModalFusion(nn.Module):
    """Fusion module for combining multiple modalities"""
    
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        
        self.config = config
        self.fusion_type = config.fusion_type
        
        if self.fusion_type == "adaptive":
            # Adaptive fusion with learned weights
            self.fusion_weights = nn.Parameter(torch.ones(5))  # 5 modalities
            self.fusion_mlp = nn.Sequential(
                nn.Linear(config.hidden_dim * 5, config.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            )
            
        elif self.fusion_type == "cross_attention":
            # Cross-attention fusion
            self.cross_modal_layers = nn.ModuleList([
                CrossModalAttention(config)
                for _ in range(3)
            ])
            
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(
        self,
        modality_features: Dict[Modality, torch.Tensor],
        primary_modality: Modality = Modality.TEXT
    ) -> torch.Tensor:
        """Fuse multiple modality features"""
        
        if self.fusion_type == "concatenate":
            # Simple concatenation
            all_features = []
            for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.CODE, Modality.MATH]:
                if modality in modality_features:
                    all_features.append(modality_features[modality])
            
            if len(all_features) == 1:
                fused = all_features[0]
            else:
                # Concatenate along sequence dimension
                fused = torch.cat(all_features, dim=1)
            
        elif self.fusion_type == "adaptive":
            # Adaptive weighted fusion
            all_features = []
            weights = F.softmax(self.fusion_weights, dim=0)
            
            for i, modality in enumerate([Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.CODE, Modality.MATH]):
                if modality in modality_features:
                    # Pool each modality
                    pooled = modality_features[modality].mean(dim=1)
                    weighted = pooled * weights[i]
                    all_features.append(weighted)
            
            if all_features:
                concatenated = torch.cat(all_features, dim=-1)
                # Pad if necessary
                if concatenated.shape[-1] < self.config.hidden_dim * 5:
                    padding = torch.zeros(
                        concatenated.shape[0],
                        self.config.hidden_dim * 5 - concatenated.shape[-1],
                        device=concatenated.device
                    )
                    concatenated = torch.cat([concatenated, padding], dim=-1)
                
                fused = self.fusion_mlp(concatenated)
                fused = fused.unsqueeze(1)  # Add sequence dimension back
            else:
                # Fallback
                fused = list(modality_features.values())[0]
            
        elif self.fusion_type == "cross_attention":
            # Cross-attention fusion
            primary_features = modality_features.get(primary_modality)
            
            if primary_features is None:
                primary_features = list(modality_features.values())[0]
            
            fused = primary_features
            for layer in self.cross_modal_layers:
                for modality, features in modality_features.items():
                    if modality != primary_modality:
                        fused = layer(fused, features)
        
        else:
            # Default: average
            all_features = list(modality_features.values())
            fused = torch.stack(all_features).mean(dim=0)
        
        # Final projection and normalization
        fused = self.output_proj(fused)
        fused = self.norm(fused)
        
        return fused


class UnifiedMultiModalModel(nn.Module):
    """Main multi-modal model combining all modalities"""
    
    def __init__(self, config: MultiModalConfig, base_model: nn.Module):
        super().__init__()
        
        self.config = config
        self.base_model = base_model
        
        # Modality embeddings
        self.modality_embedding = ModalityEmbedding(len(Modality), config.hidden_dim)
        
        # Encoders for each modality
        self.vision_encoder = VisionEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.code_encoder = CodeEncoder(config)
        self.math_encoder = MathEncoder(config)
        
        # Text uses base model embeddings
        self.text_embeddings = base_model.embed_tokens if hasattr(base_model, 'embed_tokens') else None
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(config)
        
        # Modal dropout for robustness
        self.modal_dropout = nn.Dropout(config.modal_dropout)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def encode_modality(
        self,
        modality: Modality,
        data: torch.Tensor,
        add_modality_embedding: bool = True
    ) -> torch.Tensor:
        """Encode a single modality"""
        
        if modality == Modality.TEXT:
            if self.text_embeddings is not None:
                encoded = self.text_embeddings(data)
            else:
                # Fallback
                encoded = torch.randn(data.shape[0], data.shape[1], self.config.hidden_dim)
                
        elif modality == Modality.IMAGE:
            encoded = self.vision_encoder(data)
            
        elif modality == Modality.AUDIO:
            encoded = self.audio_encoder(data)
            
        elif modality == Modality.CODE:
            encoded = self.code_encoder(data)
            
        elif modality == Modality.MATH:
            encoded = self.math_encoder(data)
            
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Add modality embedding
        if add_modality_embedding:
            modality_emb = self.modality_embedding(modality.value, encoded.shape)
            encoded = encoded + modality_emb
        
        return encoded
    
    def forward(
        self,
        inputs: Dict[Modality, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        primary_modality: Modality = Modality.TEXT,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Forward pass through multi-modal model"""
        
        # Encode each modality
        encoded_modalities = {}
        
        for modality, data in inputs.items():
            # Apply modal dropout during training
            if self.training and np.random.random() < self.config.modal_dropout:
                continue
            
            encoded = self.encode_modality(modality, data)
            encoded_modalities[modality] = encoded
        
        # Ensure at least one modality is present
        if not encoded_modalities:
            # Fallback to primary modality
            if primary_modality in inputs:
                encoded = self.encode_modality(primary_modality, inputs[primary_modality])
                encoded_modalities[primary_modality] = encoded
            else:
                # Use first available modality
                modality, data = next(iter(inputs.items()))
                encoded = self.encode_modality(modality, data)
                encoded_modalities[modality] = encoded
        
        # Fuse modalities
        fused_features = self.fusion(encoded_modalities, primary_modality)
        
        # Project
        output_features = self.output_projection(fused_features)
        
        # Pass through base model transformer layers
        if hasattr(self.base_model, 'layers'):
            for layer in self.base_model.layers:
                output_features, _ = layer(output_features)
        
        # Get logits if base model has lm_head
        logits = None
        if hasattr(self.base_model, 'lm_head'):
            if self.base_model.lm_head is not None:
                logits = self.base_model.lm_head(output_features)
            else:
                # Tied embeddings
                logits = F.linear(output_features, self.text_embeddings.weight)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None and logits is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': output_features,
                'encoded_modalities': encoded_modalities,
                'fused_features': fused_features
            }
        else:
            return logits if logits is not None else output_features
