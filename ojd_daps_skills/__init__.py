"""ojd_daps_skills."""

from pathlib import Path
from typing import Optional
import importlib.resources

import yaml
import os
from platformdirs import user_cache_path
from spacy.tokens import Doc


def get_yaml_config(file_path: Path) -> Optional[dict]:
    """Fetch yaml config and return as dict if it exists."""
    if file_path.exists():
        with open(file_path, "rt") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


PROJECT_DIR = Path(__file__).resolve().parents[1]

info_out = str(PROJECT_DIR / "info.log")
error_out = str(PROJECT_DIR / "errors.log")

_base_config_path = Path(__file__).parent.resolve() / "config/base.yaml"
config = get_yaml_config(_base_config_path)

bucket_name = "open-jobs-lake"

PUBLIC_MODEL_FOLDER_PATH = Path(
    os.environ.get("OJD_DAPS_SKILLS_CACHE", user_cache_path("ojd_daps_skills"))
).expanduser()
PACKAGE_PATH = importlib.resources.files("ojd_daps_skills")


def setup_spacy_extensions():
    if not Doc.has_extension("skill_spans"):
        Doc.set_extension("skill_spans", default=[], force=True)
    if not Doc.has_extension("mapped_skills"):
        Doc.set_extension("mapped_skills", default=[], force=True)
