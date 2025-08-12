# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from types import MethodType, ModuleType
from typing import Any, Dict, List, Type, Union

logger = logging.getLogger(__name__)

Patchable = Union[Type, ModuleType]


# Global in-memory summary for applied patches
_PATCH_SUMMARY: List[Dict[str, Any]] = []


def _qualname(obj: Any) -> str:
    """Return fully-qualified name for a class or module."""
    module_name = getattr(obj, "__module__", None)
    obj_name = getattr(obj, "__name__", repr(obj))
    return f"{module_name}.{obj_name}" if module_name else obj_name


def get_patch_summary() -> List[Dict[str, Any]]:
    """Get a copy of the applied patch summary."""
    return list(_PATCH_SUMMARY)


def print_patch_summary() -> None:
    """Print a well-formatted summary of all applied patches."""
    if not _PATCH_SUMMARY:
        msg = "[NPU Patch] No patches applied."
        print(msg)
        logger.info(msg)
        return

    lines: List[str] = []
    lines.append("\n================ NPU Patch Summary ================")
    for index, record in enumerate(_PATCH_SUMMARY, start=1):
        target = record.get("target", "<unknown>")
        patch_cls = record.get("patch_class", "<unknown>")
        lines.append(f"{index}. Target: {target}")
        lines.append(f"   Patch : {patch_cls}")
        changes: List[Dict[str, str]] = record.get("changes", [])
        if changes:
            lines.append("   Changes:")
            for change in changes:
                action = change.get("action", "?")
                kind = change.get("kind", "attr")
                name = change.get("name", "?")
                lines.append(f"     - {action:<8} {kind:<11} {name}")
        else:
            lines.append("   Changes: <none>")
    lines.append("===================================================\n")

    msg = "\n".join(lines)
    # Print to console and log for visibility in various environments
    print(msg)
    logger.info(msg)

# Copy from ArcticInference to allow patch existing classes or modules.
class NPUPatchHelper:
    """
    NPUPatchHelper provides a mechanism for cleanly patching (extending or
    modifying) existing classes or modules.

    This class uses a subscription syntax to specify the target class or
    module to be patched. Subclasses of NPUPatchHelper should define new or
    replacement attributes and methods that will be applied in-place to the
    target when `apply_patch()` is called.

    Example 1: Patching a class

    ```python
    # Define a class patch with new methods
    class ExamplePatch(NPUPatchHelper[SomeClass]):

        new_field = "This field will be added to SomeClass"

        def new_method(self):
            return "This method will be added to SomeClass"

        @classmethod
        def new_classmethod(cls):
            return "This classmethod will be added to SomeClass"

    # Apply the patch to the target class
    ExamplePatch.apply_patch()

    # Now these methods are available on the original class
    instance = SomeClass()
    instance.new_method()  # Works!
    SomeClass.new_class_method()  # Works!
    ```

    Example 2: Patching a module

    ```python
    # Define a module patch
    class ModulePatch(NPUPatchHelper[some_module]):
        NEW_CONSTANT = "This will be added to some_module"

        @staticmethod
        def new_function():
            return "This function will be added to some_module"

    ModulePatch.apply_patch()

    # The constant and function are now available in the module
    some_module.NEW_CONSTANT  # Works!
    some_module.new_function()  # Works!
    ```
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure that subclasses are created using the subscript syntax.
        if not hasattr(cls, '_npu_patch_target'):
            raise TypeError("Subclasses of NPUPatchHelper must be defined as "
                            "NPUPatchHelper[Target] to specify a patch target")

    @classmethod
    def __class_getitem__(cls, target: Patchable) -> Type:
        # The dynamic type created here will carry the target class as
        # _npu_patch_target.
        if not isinstance(target, Patchable):
            raise TypeError(f"NPUPatchHelper can only target a class or module, "
                            f"not {type(target)}")
        return type(f"{cls.__name__}[{target.__name__}]", (cls,),
                    {'_npu_patch_target': target})

    @classmethod
    def apply_patch(cls):
        """
        Patches the target class or module by replacing its attributes with
        those defined on the NPUPatchHelper subclass. Attributes are directly
        assigned to the target, and classmethods are re-bound to the target
        class before assignment.

        Raises:
            TypeError: If the NPUPatchHelper subclass is not defined with a target
                class or module.
            ValueError: If an attribute is already patched on the target.
        """
        if cls is NPUPatchHelper or not issubclass(cls, NPUPatchHelper):
            raise TypeError("apply_patch() must be called on a subclass of "
                            "NPUPatchHelper")

        target = cls._npu_patch_target

        if "_npu_patches" not in target.__dict__:
            target._npu_patches = {}

        changes: List[Dict[str, str]] = []
        for name, attr in cls.__dict__.items():

            # Skip special names and the '_npu_patch_target' itself
            if name in ("_npu_patch_target", "__dict__", "__weakref__",
                        "__module__", "__doc__", "__parameters__",):
                continue

            # Check if the attribute has already been patched
            if name in target._npu_patches:
                patch = target._npu_patches[name]
                raise ValueError(f"{target.__name__}.{name} is already "
                                 f"patched by {patch.__name__}")
            target._npu_patches[name] = cls

            # If classmethod, re-bind it to the target
            if isinstance(attr, MethodType):
                attr = MethodType(attr.__func__, target)

            # Patch the target with the new attribute
            replace = hasattr(target, name)
            setattr(target, name, attr)
            action = "replaced" if replace else "added"
            logger.info(f"{cls.__name__} {action} {target.__name__}.{name}")
            # Classify the kind of change for summary
            if isinstance(attr, classmethod):
                kind = "classmethod"
            elif isinstance(attr, staticmethod):
                kind = "staticmethod"
            elif callable(attr):
                kind = "callable"
            else:
                kind = "attribute"
            changes.append({"name": name, "action": action, "kind": kind})

        # Record a summary entry for this patch class
        _PATCH_SUMMARY.append({
            "target": _qualname(target),
            "patch_class": _qualname(cls),
            "changes": changes,
        })
