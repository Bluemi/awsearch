from pathlib import Path

from abgeordnetenwatch_python.models.questions_answers import QuestionsAnswers
from llama_cpp import Llama, ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage
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


def main_model():
    llm = Llama.from_pretrained(
        repo_id="unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF",
        filename="Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf",
        verbose=False,
        chat_format="llama-2",
        n_ctx=2048,
        n_gpu_layers=-1,
    )

    for i in range(3):
        with open(f'tests/example_texts/example0{i + 1}.txt', 'r') as file:
            text = file.read()

        answer = llm.create_chat_completion(
            messages=[
                ChatCompletionRequestSystemMessage(
                    role='system', content='Fasse den folgenden Text in zwei Sätzen zusammen.'
                ),
                ChatCompletionRequestUserMessage(role='user', content=text)
            ],
            max_tokens=512,
            seed=42,
        )

        summary = answer['choices'][0]['message']['content']
        print(summary)


if __name__ == '__main__':
    main()
