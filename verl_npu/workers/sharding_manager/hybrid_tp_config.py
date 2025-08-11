# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Dict, List, Optional

from omegaconf import DictConfig


@dataclass
class HybridTPConfig:
    """Configuration for hybrid TP strategy.
    
    This class defines the configuration for applying different TP sizes
    to different layers of the model (o_proj, mlp, lm_head).
    """
    
    # Whether to enable hybrid TP strategy
    enabled: bool = False
    
    # TP size for o_proj layer (attention output projection)
    # None means follow external tensor_model_parallel_size
    o_proj_tp_size: Optional[int] = None
    
    # TP size for MLP layers (gate_proj, up_proj, down_proj)
    # None means follow external tensor_model_parallel_size
    mlp_tp_size: Optional[int] = None
    
    # TP size for lm_head layer
    # None means follow external tensor_model_parallel_size
    lm_head_tp_size: Optional[int] = None
    
    # Custom layer name mappings for non-standard models
    # Format: {"original_layer_name": "standard_layer_name"}
    custom_layer_mappings: Optional[Dict[str, List[str]]] = None

    # Tp size from rollout config
    external_tp_size: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
    
    def _validate_custom_mappings(self):
            """validate custom mappings"""
            required_layers = ["attention_output", "mlp", "lm_head"]
            for layer in required_layers:
                if layer not in self.custom_layer_mappings:
                    raise ValueError(f"Custom layer mappings must include '{layer}'")
                

    @classmethod
    def from_dict_config(cls, config: DictConfig, external_tp_size: int) -> "HybridTPConfig":
        """Create HybridTPConfig from DictConfig.
        
        Args:
            config: DictConfig containing hybrid_tp configuration
            external_tp_size: External tensor_model_parallel_size
            
        Returns:
            HybridTPConfig instance
        """
        if not config or not config.get("enabled", False):
            return cls(enabled=False)

        # Get TP sizes, use external_tp_size as default if not specified
        o_proj_tp_size = config.get("o_proj_tp_size")
        if o_proj_tp_size is None:
            o_proj_tp_size = external_tp_size
            
        mlp_tp_size = config.get("mlp_tp_size")
        if mlp_tp_size is None:
            mlp_tp_size = external_tp_size
            
        lm_head_tp_size = config.get("lm_head_tp_size")
        if lm_head_tp_size is None:
            lm_head_tp_size = external_tp_size
        
        custom_layer_mappings = config.get("custom_layer_mappings")

        return cls(
            enabled=True,
            o_proj_tp_size=o_proj_tp_size,
            mlp_tp_size=mlp_tp_size,
            lm_head_tp_size=lm_head_tp_size,
            external_tp_size=external_tp_size,
            custom_layer_mappings=custom_layer_mappings,
        )
    


    def validate(self) -> bool:
            """Validate config fields correctness"""
            if not self.enabled:
                return True

            if not self.is_hybrid_enabled():
               return True
            
            # basic tp size check
            for tp_size in [self.o_proj_tp_size, self.mlp_tp_size, self.lm_head_tp_size]:
                if tp_size is not None:
                    if tp_size <= 0:
                        raise ValueError(f"TP size must be positive, got {tp_size}")
                    
            # 检查自定义层映射
            if self.custom_layer_mappings:
                self._validate_custom_mappings()
            
            return True
    
    def get_tp_size_for_layer(self, layer_name: str) -> int:
        """Get TP size for a specific layer.
        
        Args:
            layer_name: Name of the layer
            external_tp_size: External tensor_model_parallel_size
            
        Returns:
            TP size for the layer
        """
        if not self.enabled:
            return self.external_tp_size
        
        # Apply custom layer mappings
        mapped_name = self.custom_layer_mappings.get(layer_name, layer_name)
        
        # Determine TP size based on layer type
        if "o_proj" in mapped_name or "self_attn" in mapped_name:
            return self.o_proj_tp_size
        elif any(mlp_layer in mapped_name for mlp_layer in ["gate_proj", "up_proj", "down_proj", "mlp"]):
            return self.mlp_tp_size
        elif "lm_head" in mapped_name or "output_layer" in mapped_name:
            return self.lm_head_tp_size
        else:
            return self.external_tp_size
        
    def get_tp_size_for_layer_type(self, layer_type: str) -> int:
        """Get tp size for layer type"""
        size_map = {
            "attention_output": self.o_proj_tp_size,
            "mlp": self.mlp_tp_size,
            "lm_head": self.lm_head_tp_size
        }
        return size_map.get(layer_type) or self.external_tp_size
    
    def is_hybrid_enabled(self) -> bool:
        """Check if hybrid TP is enabled and at least one layer has different TP size."""
        if not self.enabled:
            return False
        
        # Check if at least one layer has different TP size
        # This will be checked against external_tp_size in the processor
        return any([
            self.o_proj_tp_size is not None,
            self.mlp_tp_size is not None,
            self.lm_head_tp_size is not None
        ])