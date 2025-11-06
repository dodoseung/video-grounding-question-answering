import importlib.util
import sys


def import_file(module_name, file_path, make_importable: bool = False):
    """Import a module from file path dynamically (Python 3)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


