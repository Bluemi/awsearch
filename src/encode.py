import warnings
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as functional
from abgeordnetenwatch_python.models.politician_dossier import PoliticianDossier
from transformers import AutoModel, AutoTokenizer

from tqdm import tqdm

from data import LoadDossiers

DATA_DIR = Path('data/json/bundestag')


def main():
    dossiers = LoadDossiers(DATA_DIR)
    encoder = Encoder()
    for path, dossier in tqdm(dossiers):
        encode_dossier(dossier, path, encoder)


class Encoder:
    def __init__(self):
        self.model_name = 'Alibaba-NLP/gte-multilingual-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, device_map='cuda')

    def __call__(self, texts: List[str], dimensions: int = 768, normalize: bool = True):
        with torch.no_grad():
            batch_dict = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
            batch_dict = batch_dict.to(self.model.device)
            outputs = self.model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0][:dimensions]
            if normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=1)
            return embeddings


def encode_dossier(dossier: PoliticianDossier, path: Path, encoder: Encoder):
    out_dir = Path('data/embeddings/gte/bundestag')
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_out_path = out_dir / path.relative_to(DATA_DIR).with_suffix('')

    question_embeddings = []
    answer_embeddings = []
    for index, qa in enumerate(dossier.questions_answers.questions_answers):
        if qa.question:
            question = qa.question
            if qa.question_addition:
                question += '\n' + qa.question_addition
            emb = encoder([question])
            question_embeddings.append(emb.cpu().numpy().reshape(-1))
        else:
            warnings.warn(f'missing question in {path}')

        if qa.answer:
            emb = encoder([qa.answer])
            answer_embeddings.append(emb.cpu().numpy().reshape(-1))

    question_embeddings = np.array(question_embeddings)
    answer_embeddings = np.array(answer_embeddings)

    np.savez(
        embedding_out_path,
        question_embeddings=question_embeddings,
        answer_embeddings=answer_embeddings
    )


if __name__ == '__main__':
    # main_model()
    main()
