import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from abgeordnetenwatch_python.models.questions_answers import QuestionsAnswers
from tqdm import tqdm

from data import load_dossiers


def main():
    data_dir = Path('data/json/bundestag')
    dossiers = load_dossiers(data_dir, limit=-1)
    num_questions = 0
    num_answers = 0
    for dossier in tqdm(dossiers):
        for qa in dossier.questions_answers.questions_answers:
            qa: QuestionsAnswers
            if qa.answer:
                num_answers += 1
            if qa.question:
                num_questions += 1
    print('num_questions:', num_questions)
    print('num_answers:', num_answers)


class Encoder:
    def __init__(self):
        self.model_name = 'Alibaba-NLP/gte-multilingual-base'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

    def __call__(self, texts: List[str], dimensions: int = 768, normalize: bool = True):
        with torch.no_grad():
            batch_dict = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0][:dimensions]
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings


def main_model():
    input_texts = [
        "Sehr geehrter Herr Mosblech,bitte legen Sie in Stichworten dar, wie Sie zukünftig Einfluss darauf nehmen wollen, dass rechtsextreme Parteien nicht mehr in Duisburg demonstrieren können.Mit freundlichen Grüßen Torsten Rox",
    ]

    # Tokenize the input texts
    encoder = Encoder()
    start_time = time.time()
    embeddings = encoder(input_texts)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    scores = (embeddings @ embeddings.T) * 100
    print(scores)


if __name__ == '__main__':
    main_model()
