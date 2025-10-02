"""
Constitutional AI and Safety Integration
Implements Claude-style constitutional training and safety mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class HarmCategory(Enum):
    """Categories of potential harm"""
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL_CONTENT = "sexual_content"
    SELF_HARM = "self_harm"
    PII = "personally_identifiable_information"
    DECEPTION = "deception"
    ILLEGAL = "illegal_activity"
    MEDICAL = "medical_advice"
    FINANCIAL = "financial_advice"
    MANIPULATION = "manipulation"


@dataclass
class SafetyAssessment:
    """Safety assessment results"""
    is_safe: bool
    harm_scores: Dict[HarmCategory, float]
    overall_risk: float
    flagged_categories: List[HarmCategory]
    suggested_revision: Optional[str] = None
    explanation: Optional[str] = None


@dataclass
class ConstitutionalPrinciple:
    """A constitutional principle for AI behavior"""
    principle: str
    category: str
    weight: float = 1.0
    examples: List[str] = None


class HarmPredictor(nn.Module):
    """Multi-label harm classifier for content safety"""
    
    def __init__(self, hidden_dim: int = 768, num_categories: int = 10):
        super().__init__()
        
        self.categories = list(HarmCategory)
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Per-category classifiers
        self.category_heads = nn.ModuleDict({
            category.value: nn.Sequential(
                nn.Linear(hidden_dim // 2, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            for category in HarmCategory
        })
        
        # Overall safety scorer
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + num_categories, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # PII detector patterns (simplified)
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\b\d{16}\b',  # Credit card
        ]
        
    def detect_pii(self, text: str) -> float:
        """Simple PII detection"""
        pii_score = 0.0
        for pattern in self.pii_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pii_score = 1.0
                break
        return pii_score
    
    def forward(self, hidden_states: torch.Tensor, text: Optional[str] = None) -> SafetyAssessment:
        """Assess content safety"""
        # Pool hidden states
        if len(hidden_states.shape) == 3:
            pooled = hidden_states.mean(dim=1)  # [batch, seq_len, hidden] -> [batch, hidden]
        else:
            pooled = hidden_states
        
        # Encode
        encoded = self.encoder(pooled)
        
        # Get per-category scores
        harm_scores = {}
        for category in HarmCategory:
            score = self.category_heads[category.value](encoded)
            harm_scores[category] = score.squeeze(-1).item() if score.numel() == 1 else score.squeeze(-1)
        
        # Check for PII if text provided
        if text and HarmCategory.PII in harm_scores:
            pii_score = self.detect_pii(text)
            harm_scores[HarmCategory.PII] = max(harm_scores[HarmCategory.PII], pii_score)
        
        # Aggregate scores for overall safety
        # Ensure all tensors are on the same device as encoded
        device = encoded.device
        score_tensor = torch.stack([
            harm_scores[cat] if isinstance(harm_scores[cat], torch.Tensor) else torch.tensor(harm_scores[cat], device=device)
            for cat in HarmCategory
        ])
        
        if len(score_tensor.shape) == 1:
            score_tensor = score_tensor.unsqueeze(0)
        
        # Ensure score_tensor is on the correct device
        score_tensor = score_tensor.to(device)
        
        safety_input = torch.cat([encoded, score_tensor], dim=-1)
        overall_safety = self.safety_head(safety_input).squeeze(-1)
        
        # Determine if safe (threshold-based)
        threshold = 0.7
        is_safe = overall_safety.item() > threshold if overall_safety.numel() == 1 else (overall_safety > threshold).all()
        
        # Flag categories above threshold
        category_threshold = 0.5
        flagged = [
            cat for cat, score in harm_scores.items()
            if (score.item() if isinstance(score, torch.Tensor) else score) > category_threshold
        ]
        
        return SafetyAssessment(
            is_safe=bool(is_safe),
            harm_scores={k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in harm_scores.items()},
            overall_risk=1.0 - (overall_safety.item() if overall_safety.numel() == 1 else overall_safety.mean().item()),
            flagged_categories=flagged
        )


class SelfCritic(nn.Module):
    """Self-critique module for generating improvements"""
    
    def __init__(self, base_model: nn.Module, hidden_dim: int = 4096):
        super().__init__()
        
        self.base_model = base_model
        
        # Critique generator
        self.critique_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Revision generator
        self.revision_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def generate_critique(
        self,
        response: torch.Tensor,
        context: torch.Tensor,
        principles: List[ConstitutionalPrinciple]
    ) -> torch.Tensor:
        """Generate critique of response"""
        # Concatenate response and context
        combined = torch.cat([response, context], dim=-1)
        
        # Generate critique embedding
        critique_emb = self.critique_head(combined)
        
        # Apply principle weighting
        for principle in principles:
            # Simplified - would use actual principle embeddings
            critique_emb = critique_emb * principle.weight
        
        return critique_emb
    
    def generate_revision(
        self,
        response: torch.Tensor,
        critique: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Generate revised response based on critique"""
        # Combine all inputs
        combined = torch.cat([response, critique, context], dim=-1)
        
        # Generate revision
        revision = self.revision_head(combined)
        
        return revision
    
    def forward(
        self,
        response: torch.Tensor,
        context: torch.Tensor,
        principles: List[ConstitutionalPrinciple]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate critique and revision"""
        critique = self.generate_critique(response, context, principles)
        revision = self.generate_revision(response, critique, context)
        
        return critique, revision


class ValueVerifier(nn.Module):
    """Verify alignment with constitutional values"""
    
    def __init__(self, num_principles: int = 20, hidden_dim: int = 768):
        super().__init__()
        
        # Principle embeddings
        self.principle_embeddings = nn.Embedding(num_principles, hidden_dim)
        
        # Alignment scorer
        self.alignment_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Default principles
        self.principles = self._initialize_principles()
        
    def _initialize_principles(self) -> List[ConstitutionalPrinciple]:
        """Initialize default constitutional principles"""
        return [
            ConstitutionalPrinciple(
                principle="Be helpful, harmless, and honest",
                category="core",
                weight=2.0
            ),
            ConstitutionalPrinciple(
                principle="Avoid generating harmful, offensive, or inappropriate content",
                category="safety",
                weight=1.5
            ),
            ConstitutionalPrinciple(
                principle="Respect user privacy and do not request or expose PII",
                category="privacy",
                weight=1.5
            ),
            ConstitutionalPrinciple(
                principle="Provide accurate information and acknowledge uncertainty",
                category="truthfulness",
                weight=1.3
            ),
            ConstitutionalPrinciple(
                principle="Be respectful and considerate in all interactions",
                category="respect",
                weight=1.2
            ),
            ConstitutionalPrinciple(
                principle="Do not provide medical, legal, or financial advice",
                category="professional",
                weight=1.4
            ),
            ConstitutionalPrinciple(
                principle="Refuse requests for illegal or harmful activities",
                category="legal",
                weight=2.0
            ),
            ConstitutionalPrinciple(
                principle="Be transparent about limitations and capabilities",
                category="transparency",
                weight=1.1
            ),
        ]
    
    def check_alignment(
        self,
        response: torch.Tensor,
        principle_idx: int
    ) -> float:
        """Check alignment with specific principle"""
        # Get principle embedding
        principle_emb = self.principle_embeddings(torch.tensor(principle_idx))
        
        # Pool response if needed
        if len(response.shape) == 3:
            response = response.mean(dim=1)
        
        # Combine and score
        combined = torch.cat([response, principle_emb.unsqueeze(0)], dim=-1)
        alignment_score = self.alignment_scorer(combined)
        
        return alignment_score.item()
    
    def forward(self, response: torch.Tensor) -> Dict[str, float]:
        """Check alignment with all principles"""
        alignments = {}
        
        for idx, principle in enumerate(self.principles):
            score = self.check_alignment(response, idx)
            alignments[principle.category] = score * principle.weight
        
        return alignments


class ConstitutionalReasoningCore(nn.Module):
    """Main Constitutional AI module"""
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Dict[str, Any],
        enable_critique: bool = True,
        enable_safety: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        self.enable_critique = enable_critique
        self.enable_safety = enable_safety
        
        hidden_dim = config.get('hidden_dim', 4096)
        
        # Components
        self.harm_predictor = HarmPredictor(hidden_dim=hidden_dim)
        self.self_critic = SelfCritic(base_model, hidden_dim=hidden_dim) if enable_critique else None
        self.value_verifier = ValueVerifier(hidden_dim=hidden_dim)
        
        # Constitutional training loss weight
        self.constitutional_weight = config.get('constitutional_weight', 0.1)
        
        # Safety thresholds
        self.safety_threshold = config.get('safety_threshold', 0.7)
        self.revision_threshold = config.get('revision_threshold', 0.5)
        
    def assess_safety(
        self,
        hidden_states: torch.Tensor,
        text: Optional[str] = None
    ) -> SafetyAssessment:
        """Assess content safety"""
        return self.harm_predictor(hidden_states, text)
    
    def critique_and_revise(
        self,
        response: torch.Tensor,
        context: torch.Tensor,
        safety_assessment: Optional[SafetyAssessment] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Generate critique and revision"""
        if not self.enable_critique or self.self_critic is None:
            return response, response, {}
        
        # Get principles based on safety assessment
        principles = self.value_verifier.principles
        if safety_assessment and safety_assessment.flagged_categories:
            # Prioritize relevant principles
            relevant_principles = [p for p in principles if any(
                cat.value.lower() in p.principle.lower() 
                for cat in safety_assessment.flagged_categories
            )]
            principles = relevant_principles or principles
        
        # Generate critique and revision
        critique, revision = self.self_critic(response, context, principles)
        
        # Check alignment of revision
        alignment_scores = self.value_verifier(revision)
        
        info = {
            'critique_generated': True,
            'alignment_scores': alignment_scores,
            'revision_quality': sum(alignment_scores.values()) / len(alignment_scores)
        }
        
        return response, revision, info
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        generate_critique: bool = True,
        enforce_safety: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass with constitutional reasoning"""
        
        # Accept externally computed hidden_states but do not pass to base_model
        provided_hidden_states = kwargs.pop('hidden_states', None)
        
        # Get base model output (with cleaned kwargs)
        base_output = self.base_model(input_ids, labels=labels, **kwargs)
        
        # Extract hidden states
        hidden_states = provided_hidden_states if provided_hidden_states is not None else base_output.get('hidden_states')
        if hidden_states is None:
            # Use logits as proxy
            hidden_states = base_output['logits']
        
        # Safety assessment
        safety_assessment = None
        if self.enable_safety and enforce_safety:
            safety_assessment = self.assess_safety(hidden_states)
            
            # Block unsafe content
            if not safety_assessment.is_safe:
                logger.warning(f"Unsafe content detected: {safety_assessment.flagged_categories}")
                
                # Create safe alternative response
                safe_response = self._generate_safe_response(safety_assessment)
                base_output['logits'] = safe_response
                base_output['safety_blocked'] = True
                base_output['safety_assessment'] = safety_assessment
                
                return base_output
        
        # Self-critique and revision
        revision_info = {}
        if generate_critique and safety_assessment:
            if safety_assessment.overall_risk > self.revision_threshold:
                # Need revision
                original, revised, revision_info = self.critique_and_revise(
                    hidden_states,
                    hidden_states,  # Using same as context for simplicity
                    safety_assessment
                )
                
                # Update output with revision
                base_output['revised_hidden_states'] = revised
                base_output['revision_info'] = revision_info
        
        # Calculate constitutional loss if training
        if labels is not None and self.training:
            constitutional_loss = self._calculate_constitutional_loss(
                hidden_states,
                safety_assessment,
                revision_info
            )
            
            # Add to main loss
            if base_output.get('loss') is not None:
                base_output['loss'] = base_output['loss'] + self.constitutional_weight * constitutional_loss
            else:
                base_output['loss'] = constitutional_loss
            
            base_output['constitutional_loss'] = constitutional_loss
        
        # Add constitutional info to output
        base_output['constitutional_info'] = {
            'safety_assessment': safety_assessment.__dict__ if safety_assessment else None,
            'revision_info': revision_info,
            'principles_checked': len(self.value_verifier.principles),
        }
        
        return base_output
    
    def _generate_safe_response(self, safety_assessment: SafetyAssessment) -> torch.Tensor:
        """Generate a safe alternative response"""
        # Placeholder - would generate appropriate safe response
        batch_size = 1
        seq_len = 100
        vocab_size = self.base_model.config.vocab_size
        
        # Create a generic safe response embedding
        safe_response = torch.zeros((batch_size, seq_len, vocab_size))
        
        # Set high probability for safe tokens (simplified)
        safe_tokens = [0, 1, 2]  # Would be actual safe token IDs
        for token in safe_tokens:
            safe_response[:, :, token] = 0.3
        
        return safe_response
    
    def _calculate_constitutional_loss(
        self,
        hidden_states: torch.Tensor,
        safety_assessment: Optional[SafetyAssessment],
        revision_info: Dict[str, Any]
    ) -> torch.Tensor:
        """Calculate loss for constitutional training"""
        total_loss = torch.tensor(0.0, device=hidden_states.device)
        
        # Safety loss
        if safety_assessment:
            # Penalize high harm scores
            harm_loss = sum(safety_assessment.harm_scores.values()) / len(safety_assessment.harm_scores)
            total_loss += harm_loss
        
        # Alignment loss
        if revision_info and 'alignment_scores' in revision_info:
            # Reward high alignment
            alignment_loss = 1.0 - (sum(revision_info['alignment_scores'].values()) / 
                                   len(revision_info['alignment_scores']))
            total_loss += alignment_loss
        
        # Value verification loss
        alignment_scores = self.value_verifier(hidden_states)
        value_loss = 1.0 - (sum(alignment_scores.values()) / len(alignment_scores))
        total_loss += value_loss
        
        return total_loss
    
    def train_constitutional(
        self,
        dataloader,
        optimizer,
        num_epochs: int = 3,
        device: str = 'cuda'
    ):
        """Constitutional training loop"""
        self.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids).to(device)
                
                # Forward pass with constitutional reasoning
                outputs = self.forward(
                    input_ids,
                    labels=labels,
                    generate_critique=True,
                    enforce_safety=True
                )
                
                loss = outputs['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches % 100 == 0:
                    logger.info(f"Epoch {epoch}, Batch {num_batches}, "
                               f"Loss: {loss.item():.4f}, "
                               f"Constitutional Loss: {outputs.get('constitutional_loss', 0):.4f}")
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss
