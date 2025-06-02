import json
from pathlib import Path
from typing import List

from abgeordnetenwatch_python.models.politician_dossier import PoliticianDossier


def load_dossiers(data_dir: Path, limit: int = -1) -> List[PoliticianDossier]:
    json_files = list(data_dir.rglob("*.json"))

    if limit > 0:
        json_files = json_files[:limit]

    dossiers = []
    for path in json_files:
        with open(path, 'r') as f:
            data = json.load(f)
            dossier = PoliticianDossier.model_validate(data)
            dossiers.append(dossier)
    return dossiers
