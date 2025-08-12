# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities to inject (alias) local modules into external package namespaces at runtime.

This allows us to provide missing modules expected by upstream packages without
modifying their source files on disk.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Iterable, Optional

from .patch_util import record_patch_entry


def inject_module_alias(
    source_module_name: str,
    target_module_name: str,
    export_symbols: Optional[Iterable[str]] = None,
) -> bool:
    """
    Inject an alias module at `target_module_name` that re-exports symbols from
    `source_module_name`.

    - If the target already exists, this function returns True without changes.
    - Ensures the target is also attached to its parent package attribute so that
      `import parent.child` works reliably.

    Args:
        source_module_name: Existing module to alias from (e.g.,
            'verl_npu.workers.sharding_manager.hybrid_tp_config').
        target_module_name: Target alias module to create (e.g.,
            'verl.workers.sharding_manager.hybrid_tp_config').
        export_symbols: Optional iterable of attribute names to re-export. If
            None, exports all public attributes (without leading underscore).

    Returns:
        True if injection succeeded or already present, False on failure.
    """
    try:
        # If target module already exists, treat as success
        if target_module_name in sys.modules:
            return True

        source_mod = importlib.import_module(source_module_name)

        # Create the target module and copy selected symbols
        target_mod = ModuleType(target_module_name)
        target_mod.__package__ = target_module_name.rsplit('.', 1)[0]

        if export_symbols is None:
            export_symbols = [
                name for name in dir(source_mod) if not name.startswith('_')
            ]

        for name in export_symbols:
            setattr(target_mod, name, getattr(source_mod, name))

        # Register in sys.modules
        sys.modules[target_module_name] = target_mod

        # Attach to parent package as attribute for `import parent.child`
        if '.' in target_module_name:
            parent_name, child_name = target_module_name.rsplit('.', 1)
            parent_mod = importlib.import_module(parent_name)
            setattr(parent_mod, child_name, target_mod)

        # Record patch entry for summary
        changes = [{"name": name, "action": "added", "kind": "module_attr"} for name in export_symbols]
        record_patch_entry(
            target_obj=target_module_name,
            patch_obj=f"alias:{source_module_name}",
            changes=changes,
        )

        return True
    except Exception:
        return False


def bootstrap_default_aliases() -> None:
    """Bootstrap default module aliases required by upstream packages.

    This should be called as early as possible (e.g., package __init__) so that
    any subsequent imports that rely on these modules succeed without having to
    restructure import orders elsewhere.
    """
    package_root = __name__.split('.')[0]  # e.g., 'verl_npu'

    # Map local hybrid_tp_config to upstream verl path
    source_mod = f"{package_root}.workers.sharding_manager.hybrid_tp_config"
    target_mod = "verl.workers.sharding_manager.hybrid_tp_config"
    inject_module_alias(source_module_name=source_mod, target_module_name=target_mod)


