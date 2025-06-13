import json
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict
from itertools import islice

import numpy as np
from abgeordnetenwatch_python.models.politician_dossier import PoliticianDossier
from abgeordnetenwatch_python.models.questions_answers import QuestionAnswerResult


class LoadDossiers:
    def __init__(self, data_dir: Path, limit: int = -1):
        self.data_dir = data_dir
        json_files = data_dir.rglob("*.json")
        if limit > 0:
            json_files = islice(json_files, limit)
        self.json_files = list(json_files)

    def __iter__(self) -> Iterator[Tuple[Path, PoliticianDossier]]:
        for path in self.json_files:
            with open(path, 'r') as f:
                data = json.load(f)
                yield path, PoliticianDossier.model_validate(data)

    def __len__(self):
        return len(self.json_files)


class LoadDossierEmbeddings:
    def __init__(self, data_dir: Path, embedding_name: str, limit: int = -1):
        self.load_dossiers = LoadDossiers(data_dir, limit)
        self.embedding_name = embedding_name
        self.data_dir = data_dir

    def __iter__(self) -> Iterator[Tuple[Path, int, QuestionAnswerResult, Optional[np.ndarray], Optional[np.ndarray]]]:
        embedding_dir = Path('data') / 'embeddings' / self.embedding_name / self.data_dir.name
        for path, dossier in self.load_dossiers:
            embedding_path = embedding_dir / path.relative_to(self.data_dir).with_suffix('.npz')
            embedding = np.load(embedding_path)
            question_embeddings = embedding['question_embeddings']
            answer_embeddings = embedding['answer_embeddings']
            q_index = 0
            a_index = 0
            for i, qa in enumerate(dossier.questions_answers.questions_answers):
                question_embedding = None
                answer_embedding = None
                if qa.question:
                    question_embedding = question_embeddings[q_index]
                    q_index += 1
                if qa.answer:
                    answer_embedding = answer_embeddings[a_index]
                    a_index += 1

                yield path, i, qa, question_embedding, answer_embedding


def load_qa_id(qa_id: str, cache: Optional[Dict[str, PoliticianDossier]] = None) -> QuestionAnswerResult:
    filename, i = qa_id.split(':')
    i = int(i)

    dossier = None
    if cache is not None:
        dossier = cache.get(filename)
    if dossier is None:
        with open(f'data/json/bundestag/{filename}', 'r') as file:
            data = json.load(file)
            dossier = PoliticianDossier.model_validate(data)
    return dossier.questions_answers.questions_answers[i]
