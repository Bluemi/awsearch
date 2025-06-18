import argparse
import random
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_clusters', type=int, nargs='+')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument(
        '--ratio2d', type=float, default=0.5,
        help='How much to use the 2d embedding coordinates for clustering. '
             '0.0 means not to use it, 1.0 means only use it. Defaults to 0.5'
    )
    parser.add_argument(
        '--overview', action='store_true', help='Run in overview mode (do not calc topics, save less information)'
    )
    return parser.parse_args()


def main():
    args = get_args()

    d2_data = np.load(Path('data') / 'embeddings' / 'tsne' / 'embedding.npz')
    embeddings_2d = d2_data['embeddings_2d']
    # normalize embeddings
    if not args.overview:
        embeddings_2d = embeddings_2d / np.max(embeddings_2d, axis=0, keepdims=True)
    urls = d2_data['urls']
    url_2d_emb = {url: d2e for url, d2e in zip(urls, embeddings_2d)}

    embeddings = LoadDossierEmbeddings(Path('data') / 'json' / 'bundestag', 'gte', limit=args.limit)
    question_embeddings = []
    urls = []
    questions = []
    qa_ids = []
    for path, i, qa, question_embedding, answer_embedding in tqdm(embeddings, desc='loading data'):
        if question_embedding is not None:
            if args.overview:
                full_embeddings = url_2d_emb[qa.url]
            else:
                full_embeddings = np.concatenate((
                    question_embedding * (1.0 - args.ratio2d),
                    url_2d_emb[qa.url] * args.ratio2d
                ))
            question_embeddings.append(full_embeddings)
            questions.append(f'{qa.question}\n{qa.question_addition}')
            urls.append(qa.url)
            qa_ids.append(f'{str(path.name)}:{str(i)}')
    question_embeddings = np.array(question_embeddings)

    extractor = None
    if not args.overview:
        extractor = ClusterTopicExtractor()

    for n_clusters in args.n_clusters:
        find_clusters(question_embeddings, questions, urls, n_clusters, extractor, args.overview, qa_ids)


def get_ranking(sentence: str) -> int:
    l = len(sentence)
    if l < 50:
        return 0
    if l < 100:
        return 2
    return 1


def find_clusters(question_embeddings, questions, urls, n_clusters, extractor, overview, qa_ids):
    print('running clustering')
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, verbose=True)
    kmeans.fit(question_embeddings)
    prediction = kmeans.predict(question_embeddings)
    if overview:
        cluster_to_url_q_n_id: Dict[int, Tuple[str, str, int, str]] = {}
        for url, p, q, qa_id in zip(urls, prediction, questions, qa_ids):
            if p in cluster_to_url_q_n_id:
                current = cluster_to_url_q_n_id[p]
                if get_ranking(q) > get_ranking(current[1]):
                    cluster_to_url_q_n_id[p] = (url, q, current[2] + 1, qa_id)
                else:
                    cluster_to_url_q_n_id[p] = (current[0], current[1], current[2] + 1, current[3])
            else:
                cluster_to_url_q_n_id[p] = (url, q, 1, qa_id)

        c_data = []
        for cluster_id, (url, q, n, qa_id) in cluster_to_url_q_n_id.items():
            c_data.append({
                'url': url,
                'question': q,
                'n': n,
                'cluster_id': int(cluster_id),
                'center_point': [float(i) for i in kmeans.cluster_centers_[cluster_id]],
                'qa_id': qa_id,
            })

        with open(f'data/embeddings/cluster/bundestag/overview{n_clusters}.json', 'w') as f:
            json.dump(c_data, f, indent=2)
    else:
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
