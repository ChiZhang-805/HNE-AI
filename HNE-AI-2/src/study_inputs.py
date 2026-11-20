"""Resolve this paper's input root without depending on another paper's tree."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_configured_root = os.environ.get('HNE_STUDY_ROOT')
if _configured_root:
    _input_path = Path(_configured_root).expanduser()
    if not _input_path.is_absolute():
        raise ValueError('HNE_STUDY_ROOT must be an absolute prepared-input directory')
    SOURCE = _input_path.resolve()
else:
    SOURCE = PROJECT_ROOT
