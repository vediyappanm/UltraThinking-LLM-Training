"""
Model Quantization for Efficient Deployment

Supports multiple quantization methods:
- GPTQ: 4-bit weights quantization
- AWQ: Activation-aware weight quantization (better quality)
- GGUF: Export for llama.cpp
- BitsAndBytes: 8-bit/4-bit quantization

Reduces model size by 50-75% with minimal quality loss.
"""

import logging
import torch
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False

try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False


class ModelQuantizer:
    """
    Universal model quantization tool
    
    Supported methods:
    - GPTQ: Good balance of speed and quality
    - AWQ: Best quality, slightly slower
    - GGUF: For llama.cpp deployment
    - INT8: Fast, minimal quality loss
    
    Example:
        >>> quantizer = ModelQuantizer("path/to/model")
        >>> quantized = quantizer.quantize(method="gptq", bits=4)
        >>> quantized.save("model-gptq-4bit")
    """
    
    METHODS = {
        "gptq": "GPTQ - 4-bit weights, good quality",
        "awq": "AWQ - 4-bit weights, best quality",
        "gguf": "GGUF - For llama.cpp (2-8 bit)",
        "int8": "INT8 - 8-bit quantization",
    }
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None
    ):
        """
        Initialize quantizer
        
        Args:
            model_path: Path to model or HuggingFace model name
            tokenizer_path: Path to tokenizer (optional)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
        logger.info(f"Quantizer initialized for model: {model_path}")
    
    def quantize(
        self,
        method: str = "gptq",
        bits: int = 4,
        output_path: Optional[str] = None,
        calibration_dataset: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Quantize model using specified method
        
        Args:
            method: Quantization method (gptq, awq, gguf, int8)
            bits: Number of bits (2-8, depending on method)
            output_path: Where to save quantized model
            calibration_dataset: Dataset for calibration (optional)
            **kwargs: Method-specific parameters
            
        Returns:
            Quantized model
            
        Example:
            >>> quantized = quantizer.quantize(
            ...     method="gptq",
            ...     bits=4,
            ...     output_path="model-4bit"
            ... )
        """
        method = method.lower()
        
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from: {list(self.METHODS.keys())}")
        
        logger.info(f"Quantizing with method: {method}, bits: {bits}")
        
        if method == "gptq":
            return self.quantize_gptq(bits, output_path, calibration_dataset, **kwargs)
        elif method == "awq":
            return self.quantize_awq(bits, output_path, calibration_dataset, **kwargs)
        elif method == "gguf":
            return self.export_gguf(bits, output_path, **kwargs)
        elif method == "int8":
            return self.quantize_int8(output_path, **kwargs)
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")
    
    def quantize_gptq(
        self,
        bits: int = 4,
        output_path: Optional[str] = None,
        calibration_dataset: Optional[Any] = None,
        group_size: int = 128,
        desc_act: bool = False,
        **kwargs
    ) -> Any:
        """
        GPTQ quantization (good balance)
        
        Args:
            bits: Number of bits (2-8)
            output_path: Save path
            calibration_dataset: Calibration data
            group_size: Quantization group size
            desc_act: Use desc_act order
            
        Returns:
            Quantized model
        """
        if not GPTQ_AVAILABLE:
            raise ImportError("auto-gptq not installed. Install: pip install auto-gptq")
        
        logger.info("Starting GPTQ quantization...")
        
        # Configure quantization
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=True,
            true_sequential=True
        )
        
        # Load model
        logger.info("Loading model for quantization...")
        model = AutoGPTQForCausalLM.from_pretrained(
            self.model_path,
            quantize_config=quantize_config
        )
        
        # Quantize
        if calibration_dataset is not None:
            logger.info("Quantizing with calibration data...")
            model.quantize(calibration_dataset)
        else:
            logger.info("Quantizing without calibration (may be less accurate)...")
            model.quantize(None)
        
        # Save
        if output_path:
            logger.info(f"Saving quantized model to {output_path}")
            model.save_quantized(output_path)
        
        logger.info("GPTQ quantization complete")
        return model
    
    def quantize_awq(
        self,
        bits: int = 4,
        output_path: Optional[str] = None,
        calibration_dataset: Optional[Any] = None,
        group_size: int = 128,
        zero_point: bool = True,
        **kwargs
    ) -> Any:
        """
        AWQ quantization (best quality)
        
        Args:
            bits: Number of bits (usually 4)
            output_path: Save path
            calibration_dataset: Calibration data
            group_size: Group size
            zero_point: Use zero-point quantization
            
        Returns:
            Quantized model
        """
        if not AWQ_AVAILABLE:
            raise ImportError("autoawq not installed. Install: pip install autoawq")
        
        logger.info("Starting AWQ quantization...")
        
        # Load model
        logger.info("Loading model...")
        model = AutoAWQForCausalLM.from_pretrained(self.model_path)
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        # Configure
        quant_config = {
            "zero_point": zero_point,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": "GEMM"
        }
        
        # Quantize
        if calibration_dataset:
            logger.info("Quantizing with calibration data...")
            model.quantize(
                tokenizer,
                quant_config=quant_config,
                calib_data=calibration_dataset
            )
        else:
            logger.warning("No calibration data provided - using default")
            model.quantize(tokenizer, quant_config=quant_config)
        
        # Save
        if output_path:
            logger.info(f"Saving to {output_path}")
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)
        
        logger.info("AWQ quantization complete")
        return model
    
    def export_gguf(
        self,
        bits: int = 4,
        output_path: Optional[str] = None,
        quant_type: str = "q4_0",
        **kwargs
    ) -> str:
        """
        Export to GGUF format for llama.cpp
        
        Args:
            bits: Number of bits (2-8)
            output_path: Output file path
            quant_type: GGUF quantization type (q4_0, q8_0, etc.)
            
        Returns:
            Path to exported file
        """
        logger.info("Exporting to GGUF format...")
        
        try:
            import convert_hf_to_gguf
        except ImportError:
            raise ImportError("llama.cpp conversion tools required")
        
        if output_path is None:
            output_path = f"{self.model_path.replace('/', '_')}.gguf"
        
        # Convert
        logger.info(f"Converting {self.model_path} to GGUF...")
        convert_hf_to_gguf.main([
            self.model_path,
            "--outfile", output_path,
            "--outtype", quant_type
        ])
        
        logger.info(f"GGUF export complete: {output_path}")
        return output_path
    
    def quantize_int8(
        self,
        output_path: Optional[str] = None,
        **kwargs
    ) -> torch.nn.Module:
        """
        8-bit quantization using BitsAndBytes
        
        Args:
            output_path: Save path
            
        Returns:
            Quantized model
        """
        logger.info("Starting INT8 quantization...")
        
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        # Configure 8-bit
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        # Load quantized
        logger.info("Loading model in INT8...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        logger.info(f"Model memory: {model.get_memory_footprint() / 1e9:.2f} GB")
        
        if output_path:
            logger.info(f"Saving to {output_path}")
            model.save_pretrained(output_path)
        
        logger.info("INT8 quantization complete")
        return model
    
    def benchmark(
        self,
        model: Any,
        prompt: str = "Once upon a time",
        max_new_tokens: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark quantized model
        
        Args:
            model: Quantized model
            prompt: Test prompt
            max_new_tokens: Generation length
            
        Returns:
            Performance metrics
        """
        import time
        from transformers import AutoTokenizer
        
        logger.info("Benchmarking model...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Warmup
        _ = model.generate(**inputs, max_new_tokens=10)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = num_tokens / total_time
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        metrics = {
            "total_time_sec": total_time,
            "tokens_generated": num_tokens,
            "tokens_per_sec": tokens_per_sec,
            "memory_mb": memory_mb,
        }
        
        logger.info(f"Benchmark results: {metrics}")
        return metrics


def quick_quantize(
    model_path: str,
    method: str = "gptq",
    bits: int = 4,
    output_path: Optional[str] = None
) -> Any:
    """
    Quick quantization function
    
    Args:
        model_path: Path to model
        method: Quantization method
        bits: Number of bits
        output_path: Output path
        
    Returns:
        Quantized model
        
    Example:
        >>> model = quick_quantize("meta-llama/Llama-2-7b-hf", "gptq", 4)
    """
    quantizer = ModelQuantizer(model_path)
    return quantizer.quantize(method, bits, output_path)


# Example usage
if __name__ == "__main__":
    print("Model Quantization - Example usage:")
    print("""
    from src.models.quantization import ModelQuantizer, quick_quantize
    
    # Quick quantization
    model = quick_quantize(
        "meta-llama/Llama-2-7b-hf",
        method="gptq",
        bits=4,
        output_path="llama-2-7b-gptq-4bit"
    )
    
    # Advanced usage
    quantizer = ModelQuantizer("path/to/model")
    
    # GPTQ (good balance)
    gptq_model = quantizer.quantize(
        method="gptq",
        bits=4,
        group_size=128,
        output_path="model-gptq"
    )
    
    # AWQ (best quality)
    awq_model = quantizer.quantize(
        method="awq",
        bits=4,
        output_path="model-awq"
    )
    
    # GGUF for llama.cpp
    gguf_path = quantizer.export_gguf(
        bits=4,
        output_path="model.gguf"
    )
    
    # Benchmark
    metrics = quantizer.benchmark(gptq_model)
    print(f"Speed: {metrics['tokens_per_sec']:.1f} tokens/sec")
    print(f"Memory: {metrics['memory_mb']:.0f} MB")
    
    # Model size comparison:
    # Original (FP16): 14 GB
    # GPTQ 4-bit: 3.5 GB (75% reduction!)
    # AWQ 4-bit: 3.5 GB (75% reduction, better quality)
    # GGUF Q4_0: 3.5 GB (llama.cpp compatible)
    
    # Speed comparison:
    # Original: 20 tokens/sec
    # GPTQ: 40-50 tokens/sec (2-2.5x faster!)
    # AWQ: 35-45 tokens/sec (1.75-2.25x faster)
    """)
