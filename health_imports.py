"""Executado pelo Node no arranque para validar torch/onnxruntime/rembg/transformers."""
import json
import sys

try:
    import torch  # noqa: F401
    import onnxruntime  # noqa: F401
    import rembg  # noqa: F401
    import transformers  # noqa: F401

    print(json.dumps({"ok": True, "torch": torch.__version__}))
except Exception as exc:  # noqa: BLE001
    print(json.dumps({"ok": False, "error": str(exc)}))
    sys.exit(1)
