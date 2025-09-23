"""
ULTRATHINK: Advanced AI Training Pipeline
"""

__version__ = "1.0.0"
__author__ = "ULTRATHINK Team"

from .models.ultrathink import UltraThinkModel, UltraThinkConfig
from .models.architecture import AdvancedGPTModel, ModelConfig
from .models.dynamic_reasoning import DynamicReasoningEngine, ReasoningPath
from .models.constitutional_ai import ConstitutionalReasoningCore
from .models.moe_advanced import MoELayer, ExpertConfig
from .models.multimodal import UnifiedMultiModalModel, MultiModalConfig, Modality

__all__ = [
    "UltraThinkModel",
    "UltraThinkConfig", 
    "AdvancedGPTModel",
    "ModelConfig",
    "DynamicReasoningEngine",
    "ReasoningPath",
    "ConstitutionalReasoningCore",
    "MoELayer",
    "ExpertConfig",
    "UnifiedMultiModalModel",
    "MultiModalConfig",
    "Modality"
]
