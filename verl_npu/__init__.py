from verl_npu.module_injection import bootstrap_default_aliases

# Ensure upstream-facing module aliases exist before any other imports
bootstrap_default_aliases()

# Import plugin afterward to apply runtime patches
from verl_npu.plugin import apply_npu_plugin

apply_npu_plugin()
