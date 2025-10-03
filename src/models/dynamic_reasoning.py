"""
Dynamic Reasoning Engine (DRE) for Adaptive Multi-Path Inference
Implements Claude 4 / GPT-5 style adaptive reasoning with complexity scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReasoningPath(Enum):
    """Available reasoning paths with different compute requirements"""
    FAST = "fast"  # <100ms - cached/simple responses
    STANDARD = "standard"  # 1-5s - normal forward pass
    EXPERT = "expert"  # expert MoE path (activates experts)
    DEEP = "deep"  # 10-60s - chain-of-thought
    ULTRA_DEEP = "ultra_deep"  # minutes - recursive reasoning


@dataclass
class ComplexityFeatures:
    """Features used for complexity scoring"""
    token_length: int
    token_entropy: float
    has_math: bool
    has_code: bool
    named_entities_count: int
    syntactic_depth: float
    conversation_depth: int
    prior_failures: int = 0
    user_preference_score: float = 0.5
    use_moe: bool = False  # Whether to use MoE for this path
    domain_signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Routing decision output"""
    path: ReasoningPath
    confidence: float
    complexity_score: float
    estimated_latency_ms: float
    debug_info: Dict[str, Any] = field(default_factory=dict)


class ComplexityScorer(nn.Module):
    """Neural network for scoring input complexity"""
    
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Feature extractors
        self.text_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Domain-specific encoders
        self.math_encoder = nn.Linear(32, hidden_dim // 4)
        self.code_encoder = nn.Linear(32, hidden_dim // 4)
        
        # Complexity predictor
        self.complexity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Feature statistics
        self.register_buffer('feature_mean', torch.zeros(feature_dim))
        self.register_buffer('feature_std', torch.ones(feature_dim))
        
    def extract_features(self, text: str, tokens: torch.Tensor) -> ComplexityFeatures:
        """Extract complexity features from input"""
        # Token statistics
        token_length = len(tokens)
        
        # Calculate token entropy
        token_probs = torch.softmax(torch.randn(len(tokens)), dim=-1)  # Placeholder
        token_entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10)).item()
        
        # Domain detection
        has_math = any(symbol in text for symbol in ['=', '∫', '∑', '∂', 'sqrt', 'log'])
        has_code = any(keyword in text for keyword in ['def', 'class', 'function', '{', '}', '()', '[]'])
        
        # Named entities (simplified)
        import re
        capitals = re.findall(r'\b[A-Z][a-z]+\b', text)
        named_entities_count = len(set(capitals))
        
        # Syntactic complexity (simplified - could use actual parser)
        syntactic_depth = len(text.split('.')) * np.log(1 + len(text.split(',')))
        
        return ComplexityFeatures(
            token_length=token_length,
            token_entropy=token_entropy,
            has_math=has_math,
            has_code=has_code,
            named_entities_count=named_entities_count,
            syntactic_depth=syntactic_depth,
            conversation_depth=0  # Set by conversation manager
        )
    
    def forward(self, features: ComplexityFeatures) -> torch.Tensor:
        """Compute complexity score from features"""
        # Create feature vector
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        feature_vec = torch.tensor([
            features.token_length / 1000.0,  # Normalize
            features.token_entropy / 10.0,
            float(features.has_math),
            float(features.has_code),
            features.named_entities_count / 20.0,
            features.syntactic_depth / 100.0,
            features.conversation_depth / 10.0,
            features.prior_failures / 5.0,
            features.user_preference_score
        ], dtype=dtype, device=device).unsqueeze(0)
        
        # Pad to feature_dim
        if feature_vec.shape[1] < self.feature_mean.shape[0]:
            padding = torch.zeros((1, self.feature_mean.shape[0] - feature_vec.shape[1]), dtype=dtype, device=device)
            feature_vec = torch.cat([feature_vec, padding], dim=1)
        
        # Normalize features
        feature_vec = (feature_vec - self.feature_mean.to(dtype=dtype, device=device)) / (self.feature_std.to(dtype=dtype, device=device) + 1e-8)
        
        # Encode features
        text_features = self.text_encoder(feature_vec)
        
        # Add domain-specific features if present
        if features.has_math:
            math_features = self.math_encoder(torch.randn(1, 32, dtype=dtype, device=device))  # Placeholder
            text_features = torch.cat([text_features, math_features], dim=-1)
        
        if features.has_code:
            code_features = self.code_encoder(torch.randn(1, 32, dtype=dtype, device=device))  # Placeholder
            text_features = torch.cat([text_features, code_features], dim=-1)
        
        # Pad if necessary
        if text_features.shape[1] < 256:
            padding = torch.zeros((1, 256 - text_features.shape[1]), dtype=dtype, device=device)
            text_features = torch.cat([text_features, padding], dim=1)
        
        # Predict complexity
        complexity_score = self.complexity_head(text_features)
        
        return complexity_score.squeeze()


class RouterNetwork(nn.Module):
    """Neural router for path selection"""
    
    def __init__(self, hidden_dim: int = 4096, router_hidden: int = 1024, n_paths: int = 4):
        super().__init__()
        
        self.n_paths = n_paths
        
        # Router MLP
        self.router = nn.Sequential(
            nn.Linear(hidden_dim + 9, router_hidden),  # +9 for complexity features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(router_hidden, router_hidden // 2),
            nn.ReLU(),
            nn.Linear(router_hidden // 2, n_paths)
        )
        
        # Confidence predictor
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim + n_paths, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor, complexity_features: ComplexityFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route to appropriate path based on input"""
        batch_size = hidden_states.shape[0]
        
        # Pool hidden states
        pooled = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        
        # Create feature vector
        dtype = hidden_states.dtype
        device = hidden_states.device
        feature_vec = torch.tensor([
            complexity_features.token_length / 1000.0,
            complexity_features.token_entropy / 10.0,
            float(complexity_features.has_math),
            float(complexity_features.has_code),
            complexity_features.named_entities_count / 20.0,
            complexity_features.syntactic_depth / 100.0,
            complexity_features.conversation_depth / 10.0,
            complexity_features.prior_failures / 5.0,
            complexity_features.user_preference_score
        ], dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        # Concatenate features
        router_input = torch.cat([pooled, feature_vec], dim=-1)
        
        # Get routing probabilities
        logits = self.router(router_input)
        probs = F.softmax(logits, dim=-1)
        
        # Predict confidence
        conf_input = torch.cat([pooled, probs], dim=-1)
        confidence = self.confidence(conf_input).squeeze(-1)
        
        return probs, confidence


class DynamicReasoningEngine(nn.Module):
    """Main DRE orchestrator for adaptive inference"""
    
    def __init__(
        self,
        base_model: nn.Module,
        config: Dict[str, Any],
        fast_model: Optional[nn.Module] = None,
        enable_caching: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.fast_model = fast_model or self._create_distilled_model()
        self.config = config
        
        # Components
        self.complexity_scorer = ComplexityScorer()
        self.router = RouterNetwork(
            hidden_dim=config.get('hidden_dim', 4096),
            n_paths=len(ReasoningPath)
        )
        # Hidden-state based complexity head to avoid placeholder randomness and to vary per-input
        self.hidden_complexity_head = nn.Sequential(
            nn.Linear(config.get('hidden_dim', 4096), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
        # Caching
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Thresholds for routing (can be learned)
        self.complexity_thresholds = {
            ReasoningPath.FAST: 0.2,
            ReasoningPath.STANDARD: 0.4,
            ReasoningPath.EXPERT: 0.7,
            ReasoningPath.DEEP: 0.85,
            ReasoningPath.ULTRA_DEEP: 0.95
        }
        
        # Latency tracking
        self.latency_history = {path: [] for path in ReasoningPath}
        
        # DRE metrics tracking
        self.activation_counts = {path: 0 for path in ReasoningPath}
        self.total_activations = 0
        self.complexity_scores = []
        self.confidence_scores = []
        self.reasoning_steps = []
    
    def _create_distilled_model(self):
        """Create a smaller distilled version of the base model"""
        # Placeholder - in practice, load a pre-distilled model
        return nn.Sequential(
            nn.Linear(self.base_model.config.n_embd, 512),
            nn.ReLU(),
            nn.Linear(512, self.base_model.config.vocab_size)
        )
    
    def _check_cache(self, input_hash: str) -> Optional[torch.Tensor]:
        """Check if response is cached"""
        if not self.enable_caching:
            return None
            
        if input_hash in self.cache:
            self.cache_hits += 1
            logger.info(f"Cache hit! Hits: {self.cache_hits}, Misses: {self.cache_misses}")
            return self.cache[input_hash]
        
        self.cache_misses += 1
        return None
    
    def _fast_inference(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fast path: cached or distilled model inference"""
        # Check cache first
        input_hash = hash(input_ids.cpu().numpy().tobytes())
        cached = self._check_cache(str(input_hash))
        if cached is not None:
            return cached
        
        # Use distilled model
        if self.fast_model is not None:
            with torch.no_grad():
                embeddings = self.base_model.embed_tokens(input_ids)
                pooled = embeddings.mean(dim=1)
                output = self.fast_model(pooled)
                
                # Cache result
                if self.enable_caching:
                    self.cache[str(input_hash)] = output
                    
                return output
        
        return None
    
    def _standard_inference(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Standard path: normal forward pass"""
        return self.base_model(input_ids, **kwargs)
    
    def _deep_inference(
        self, 
        input_ids: torch.Tensor,
        max_steps: int = 10,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Deep path: chain-of-thought reasoning"""
        outputs = []
        current_input = input_ids
        
        for step in range(max_steps):
            # Generate reasoning step
            step_output = self.base_model(current_input, **kwargs)
            outputs.append(step_output)
            
            # Check if reasoning is complete (simplified)
            if self._is_reasoning_complete(step_output):
                break
            
            # Prepare next input (would include generated tokens in practice)
            current_input = input_ids  # Placeholder
        
        # Aggregate outputs
        final_output = self._aggregate_reasoning_steps(outputs)
        return final_output
    
    def _ultra_deep_inference(
        self,
        input_ids: torch.Tensor,
        max_depth: int = 5,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Ultra-deep path: recursive reasoning with self-reflection"""
        def recursive_reason(input_ids, depth):
            if depth == 0:
                return self._standard_inference(input_ids, **kwargs)
            
            # Generate initial response
            response = self._deep_inference(input_ids, **kwargs)
            
            # Self-critique (placeholder)
            critique = self._generate_critique(response)
            
            # Refine based on critique
            refined = recursive_reason(input_ids, depth - 1)
            
            return self._merge_responses(response, refined)
        
        return recursive_reason(input_ids, max_depth)
    
    def _is_reasoning_complete(self, output: Dict[str, torch.Tensor]) -> bool:
        """Check if reasoning chain is complete"""
        # Simplified - check for end token or confidence threshold
        logits = output.get('logits', None)
        if logits is not None:
            probs = F.softmax(logits[:, -1, :], dim=-1)
            max_prob = probs.max().item()
            return max_prob > 0.95  # High confidence
        return False
    
    def _aggregate_reasoning_steps(self, outputs: List[Dict]) -> Dict[str, torch.Tensor]:
        """Aggregate multiple reasoning steps"""
        # Simple averaging (can be more sophisticated)
        aggregated = {}
        for key in outputs[0].keys():
            if isinstance(outputs[0][key], torch.Tensor):
                stacked = torch.stack([o[key] for o in outputs])
                aggregated[key] = stacked.mean(dim=0)
            else:
                aggregated[key] = outputs[-1][key]  # Take last
        return aggregated
    
    def _generate_critique(self, response: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate self-critique of response"""
        # Placeholder - would use a critique model
        return torch.randn_like(response['logits'])
    
    def _merge_responses(self, response1: Dict, response2: Dict) -> Dict[str, torch.Tensor]:
        """Merge two responses"""
        merged = {}
        for key in response1.keys():
            if isinstance(response1[key], torch.Tensor):
                # Weighted average
                merged[key] = 0.6 * response1[key] + 0.4 * response2[key]
            else:
                merged[key] = response1[key]
        return merged
    
    def route(
        self,
        input_ids: torch.Tensor,
        text: str = "",
        use_soft_routing: bool = False,
        override_path: Optional[ReasoningPath] = None
    ) -> RoutingDecision:
        """Decide which reasoning path to use"""
        
        # Extract features
        features = self.complexity_scorer.extract_features(text, input_ids[0])
        
        # Get complexity score - combine hidden-state signal with features for better variation
        # Use base embeddings as input signal but DETACH to avoid training the base model from DRE aux loss
        embeddings = self.base_model.embed_tokens(input_ids).detach()
        pooled = embeddings.mean(dim=1)  # [batch, hidden_dim]
        complexity_hidden = self.hidden_complexity_head(pooled).squeeze(-1)  # [batch]
        complexity_features = self.complexity_scorer(features).squeeze()
        # Blend signals; if batch, average feature score across batch for stability
        if isinstance(complexity_features, torch.Tensor) and complexity_features.dim() == 0:
            complexity_features_tensor = complexity_features
        else:
            # Coerce to tensor on the right device/dtype
            complexity_features_tensor = torch.as_tensor(complexity_features, dtype=complexity_hidden.dtype, device=complexity_hidden.device)
        complexity_score_tensor = 0.7 * complexity_hidden + 0.3 * complexity_features_tensor
        complexity_score = float(complexity_score_tensor.mean().detach().cpu().item())
        
        # Get router prediction (allow grads for router so it can learn via aux loss)
        probs, confidence = self.router(embeddings, features)
        
        # Override if specified
        if override_path:
            return RoutingDecision(
                path=override_path,
                confidence=1.0,
                complexity_score=complexity_score,
                estimated_latency_ms=self._estimate_latency(override_path),
                debug_info={'override': True}
            )
        
        # Soft routing: combine outputs from multiple paths
        if use_soft_routing:
            # Return probabilities for weighted combination
            probs_np = probs.detach().to(torch.float32).cpu().numpy()
            return RoutingDecision(
                path=ReasoningPath.STANDARD,  # Default
                confidence=confidence.item(),
                complexity_score=complexity_score,
                estimated_latency_ms=self._estimate_latency_weighted(probs),
                debug_info={'probs': probs_np, 'soft_routing': True}
            )
        
        # Hard routing: select single path
        path_idx = probs.argmax(dim=-1).item()
        selected_path = list(ReasoningPath)[path_idx]
        
        # Apply complexity threshold override only when NOT training
        # During training, allow the router to learn the mapping; rely on thresholds at inference time
        if not self.training:
            if complexity_score < self.complexity_thresholds[ReasoningPath.FAST]:
                selected_path = ReasoningPath.FAST
            elif complexity_score < self.complexity_thresholds[ReasoningPath.STANDARD]:
                selected_path = ReasoningPath.STANDARD
            elif complexity_score < self.complexity_thresholds[ReasoningPath.DEEP]:
                selected_path = ReasoningPath.DEEP
            elif complexity_score >= self.complexity_thresholds[ReasoningPath.ULTRA_DEEP]:
                selected_path = ReasoningPath.ULTRA_DEEP
        
        # Stash tensors for aux loss computation during forward()
        self._last_router_tensors = {
            'probs': probs,  # [batch, n_paths]
            'confidence': confidence,  # [batch]
            'complexity': complexity_score_tensor,  # [batch]
        }
        probs_np = probs.detach().to(torch.float32).cpu().numpy()
        return RoutingDecision(
            path=selected_path,
            confidence=confidence.item(),
            complexity_score=complexity_score,
            estimated_latency_ms=self._estimate_latency(selected_path),
            debug_info={
                'probs': probs_np,
                'features': features.__dict__
            }
        )
    
    def _estimate_latency(self, path: ReasoningPath) -> float:
        """Estimate latency for a given path"""
        latency_ranges = {
            ReasoningPath.FAST: (10, 100),
            ReasoningPath.STANDARD: (1000, 5000),
            ReasoningPath.DEEP: (10000, 60000),
            ReasoningPath.ULTRA_DEEP: (60000, 300000)
        }
        
        if self.latency_history[path]:
            # Use historical average
            return np.mean(self.latency_history[path][-10:])
        
        # Use midpoint of range
        min_lat, max_lat = latency_ranges[path]
        return (min_lat + max_lat) / 2
    
    def _estimate_latency_weighted(self, probs: torch.Tensor) -> float:
        """Estimate weighted latency for soft routing"""
        latencies = [self._estimate_latency(path) for path in ReasoningPath]
        weighted_latency = sum(p * l for p, l in zip(probs[0].detach().to(torch.float32).cpu().numpy(), latencies))
        return weighted_latency
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current DRE metrics for logging"""
        if self.total_activations == 0:
            return {
                'activation_rate': 0.0,
                'avg_complexity': 0.0,
                'avg_confidence': 0.0,
                'avg_reasoning_steps': 0.0,
                'path_distribution': {path.value: 0.0 for path in ReasoningPath}
            }
        
        # Calculate activation rates per path
        path_distribution = {
            path.value: self.activation_counts[path] / self.total_activations * 100
            for path in ReasoningPath
        }
        
        # Calculate averages
        avg_complexity = float(np.mean(self.complexity_scores[-100:])) if self.complexity_scores else 0.0
        avg_confidence = float(np.mean(self.confidence_scores[-100:])) if self.confidence_scores else 0.0
        avg_reasoning_steps = float(np.mean(self.reasoning_steps[-50:])) if self.reasoning_steps else 0.0
        
        # Cache efficiency
        cache_hit_rate = 0.0
        if self.enable_caching and (self.cache_hits + self.cache_misses) > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
        
        return {
            'activation_rate': self.total_activations,
            'avg_complexity': avg_complexity,
            'avg_confidence': avg_confidence,
            'avg_reasoning_steps': avg_reasoning_steps,
            'path_distribution': path_distribution,
            'cache_hit_rate': cache_hit_rate,
            'total_cache_hits': self.cache_hits,
            'total_cache_misses': self.cache_misses
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        text: str = "",
        override_path: Optional[ReasoningPath] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Main forward pass with dynamic routing"""
        
        # Route to appropriate path
        routing_decision = self.route(input_ids, text, override_path=override_path)
        
        # Track timing
        start_time = time.time()
        
        # Execute selected path
        if routing_decision.path == ReasoningPath.FAST:
            output = self._fast_inference(input_ids, **kwargs)
            # Convert to standard format if needed
            if not isinstance(output, dict):
                output = {'logits': output}
                
        elif routing_decision.path == ReasoningPath.STANDARD:
            output = self._standard_inference(input_ids, **kwargs)
            
        elif routing_decision.path == ReasoningPath.EXPERT:
            # Expert path shares the same base forward; UltraThinkCore will apply MoE based on routing_info['use_moe']
            output = self._standard_inference(input_ids, **kwargs)
            
        elif routing_decision.path == ReasoningPath.DEEP:
            output = self._deep_inference(input_ids, **kwargs)
            
        elif routing_decision.path == ReasoningPath.ULTRA_DEEP:
            output = self._ultra_deep_inference(input_ids, **kwargs)
            
        else:
            raise ValueError(f"Unknown reasoning path: {routing_decision.path}")
        
        # Track latency
        latency_ms = (time.time() - start_time) * 1000
        self.latency_history[routing_decision.path].append(latency_ms)
        
        # Update DRE metrics
        self.activation_counts[routing_decision.path] += 1
        self.total_activations += 1
        self.complexity_scores.append(routing_decision.complexity_score)
        self.confidence_scores.append(routing_decision.confidence)
        
        # Compute a small auxiliary loss to train the router (balance + latency + confidence)
        dre_aux_loss = None
        try:
            if self.training and hasattr(self, '_last_router_tensors'):
                probs = self._last_router_tensors['probs']  # [batch, n_paths]
                confidence = self._last_router_tensors['confidence']  # [batch]
                # Encourage balanced usage across paths (Switch-Transformer style)
                target_uniform = torch.full_like(probs[0], 1.0 / probs.shape[-1])
                balance_loss = (probs.mean(dim=0) - target_uniform).pow(2).mean()
                # Penalize expected latency (prefer cheaper paths unless LM loss demands otherwise)
                # Relative costs for FAST, STANDARD, DEEP, ULTRA_DEEP
                path_costs = torch.tensor([0.1, 1.0, 2.0, 3.0], dtype=probs.dtype, device=probs.device)
                expected_cost = (probs * path_costs).sum(dim=-1).mean()
                # Encourage higher confidence
                conf_loss = -torch.log(confidence.clamp_min(1e-6)).mean()
                dre_aux_loss = balance_loss + 0.1 * expected_cost + 0.01 * conf_loss
        except Exception:
            dre_aux_loss = None
        
        # Track reasoning steps for deep paths
        if routing_decision.path in [ReasoningPath.DEEP, ReasoningPath.ULTRA_DEEP]:
            steps = routing_decision.debug_info.get('reasoning_steps', 1)
            self.reasoning_steps.append(steps)
        
        # Add routing info to output
        output['routing_info'] = {
            'path': routing_decision.path.value,
            'complexity_score': routing_decision.complexity_score,
            'confidence': routing_decision.confidence,
            'latency_ms': latency_ms,
            'debug': routing_decision.debug_info,
            'dre_metrics': self.get_current_metrics(),
            'use_moe': (routing_decision.path == ReasoningPath.EXPERT)
        }
        # Expose aux loss to the trainer for joint optimization
        if dre_aux_loss is not None:
            output['dre_aux_loss'] = dre_aux_loss
        
        # Avoid issues with torch.compile/torch._dynamo tracing Python f-strings and time
        try:
            is_compiling = getattr(torch._dynamo, 'is_compiling', lambda: False)()
        except Exception:
            is_compiling = False
        if not is_compiling:
            # Use logger parameter interpolation to avoid formatting issues
            logger.info("DRE: Path=%s, Complexity=%.3f, Latency=%.1fms",
                        routing_decision.path.value,
                        float(routing_decision.complexity_score),
                        float(latency_ms))

        return output
