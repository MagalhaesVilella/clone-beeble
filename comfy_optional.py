"""
Integração opcional com ComfyUI (Depth / IC-Light / deflicker).

Definir SWITCHX_COMFYUI_URL (ex.: http://127.0.0.1:8188) e colocar workflows JSON
em ComfyUI/user_workflows/ — este módulo apenas documenta o gancho; não há
chamadas HTTP activas por defeito para não bloquear o MVP.
"""
from __future__ import annotations

import os
from typing import Any, Optional


def comfyui_base_url() -> Optional[str]:
    u = os.environ.get("SWITCHX_COMFYUI_URL", "").strip()
    return u or None


def run_workflow_if_configured(_workflow_name: str, _payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    Reserva para POST /prompt no ComfyUI. Sem URL configurada devolve None.
    """
    if not comfyui_base_url():
        return None
    # Implementação deliberadamente omitida: evita dependência de rede no worker de máscaras.
    return None
