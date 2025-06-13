import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List

from llama_cpp import Llama, ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from data import LoadDossierEmbeddings


class ClusterTopicExtractor:
    def __init__(self):
        self.llm = Llama.from_pretrained(
            repo_id="unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF",
            filename="Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf",
            verbose=False,
            chat_format="llama-2",
            n_ctx=2048*8,
            n_gpu_layers=-1,
        )

    def __call__(self, texts: List[str]):
        content = '\n'.join(texts)

        answer = self.llm.create_chat_completion(
            messages=[
                ChatCompletionRequestSystemMessage(
                    role='system',
                    content='Alle folgenden Texte haben ein gemeinsames politisches Thema / eine politische Richtung. '
                            'Nenne dieses Thema/Richtung. Antworte nur mit einem oder zwei Wörtern. '
                            'Vermeide das Wort "Politik".'
                ),
                ChatCompletionRequestUserMessage(role='user', content=content)
            ],
            max_tokens=512,
            seed=42,
        )

        return answer['choices'][0]['message']['content']


def main():
    embeddings = LoadDossierEmbeddings(Path('data') / 'json' / 'bundestag', 'gte', limit=-1)
    question_embeddings = []
    urls = []
    questions = []
    for path, i, qa, question_embedding, answer_embedding in tqdm(embeddings, desc='loading data'):
        if question_embedding is not None:
            question_embeddings.append(question_embedding)
            questions.append(f'{qa.question}\n{qa.question_addition}')
            urls.append(qa.url)
    question_embeddings = np.array(question_embeddings)

    extractor = ClusterTopicExtractor()
    for n_clusters in [25, 35, 50, 75, 100, 150, 250, 500]:
        find_clusters(question_embeddings, questions, urls, n_clusters, extractor)


def find_clusters(question_embeddings, questions, urls, n_clusters, extractor):
    print('running clustering')
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, verbose=True)
    kmeans.fit(question_embeddings)
    prediction = kmeans.predict(question_embeddings)
    url_to_cluster = {url: int(p) for url, p in zip(urls, prediction)}
    with open(f'data/embeddings/cluster/bundestag/cluster{n_clusters}.json', 'w') as f:
        json.dump(url_to_cluster, f, indent=2)
    cluster_to_questions = defaultdict(list)
    for cluster, q in zip(prediction, questions):
        cluster_to_questions[int(cluster)].append(q)
    cluster_to_topic = {}
    for cluster_id, questions in tqdm(cluster_to_questions.items(), desc='finding topics'):
        questions = random.choices(questions, k=10)
        topic = extractor(questions)
        cluster_to_topic[cluster_id] = topic
    with open(f'data/embeddings/cluster/bundestag/topics{n_clusters}.json', 'w') as f:
        json.dump(cluster_to_topic, f, indent=2)


if __name__ == '__main__':
    main()
