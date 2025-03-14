import pandas as pd
from tqdm import tqdm
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
from langchain.prompts import PromptTemplate
import time


# Функции для вычисления метрик
def compute_exact_match(prediction, ground_truth):
    """Возвращает 1, если предсказание точно совпадает с эталоном (после удаления лишних пробелов), иначе 0."""
    return int(prediction.strip() == ground_truth.strip())

def compute_f1(prediction, ground_truth):
    """Вычисляет F1-меру на уровне токенов."""
    pred_tokens = prediction.strip().split()
    true_tokens = ground_truth.strip().split()
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return 0.0
    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_bertscore(prediction, ground_truth, lang="en"):
    """
    Вычисляет BERTScore для пары текстов.
    Функция bert_score.score возвращает кортеж (P, R, F1) для списка предсказаний и списка эталонов.
    Возвращаемое значение – среднее значение F1 (BERTScore).
    """
    P, R, F1 = bertscore([prediction], [ground_truth], lang=lang, verbose=False)
    return F1[0].item()

def compute_bleu(prediction, ground_truth):
    """
    Вычисляет BLEU для одного предложения.
    Используем сглаживание, чтобы избежать проблем при малом количестве n-грамм.
    """
    pred_tokens = prediction.strip().split()
    true_tokens = ground_truth.strip().split()
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([true_tokens], pred_tokens, smoothing_function=smoothie)
    return bleu

# Функция для оценки промптов и сохранения результатов в Excel
def evaluate_prompts(excel_file: str, save_file: str, rag, delay: int = 10, bert_lang: str = "ru"):
    """
    Загружает Excel-файл с данными, для каждого запроса (из столбца "Вопрос")
    генерирует ответ с помощью RAG-системы, сравнивает с эталоном (столбец "Ответ") по метрикам EM, F1, BERTScore, BLEU,
    и сохраняет результаты в Excel с вычислением среднего по всем парам.
    
    Параметры:
      excel_file: путь до исходного Excel-файла с данными.
      save_file: путь до файла для сохранения результатов (Excel).
      rag: экземпляр RAGSystem.
      delay: задержка между запросами (в секундах).
      bert_lang: язык для расчёта BERTScore ("ru" для русского).
    """
    # Загружаем Excel
    df = pd.read_excel(excel_file)[:10]
    
    # Проверяем наличие необходимых столбцов
    if "Вопрос" not in df.columns or "Ответ" not in df.columns:
        raise ValueError("В Excel файле должны быть столбцы 'Вопрос' и 'Ответ'.")
    
    results = []
    # Итерация с progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Оценка промптов"):
        query = row["Вопрос"]
        ground_truth = row["Ответ"]
        
        # Получаем финальный промпт (для отображения)
        prompt = rag.generate_prompt(query)
        # Получаем ответ модели
        prediction = rag.get_final_answer(query)
        
        # Вычисляем метрики
        em = compute_exact_match(prediction, ground_truth)
        f1 = compute_f1(prediction, ground_truth)
        bert_f1 = compute_bertscore(prediction, ground_truth, lang=bert_lang)
        bleu = compute_bleu(prediction, ground_truth)
        
        results.append({
            "Вопрос": query,
            "Эталонный Ответ": ground_truth,
            "Промпт": prompt,
            "Ответ Модели": prediction,
            "Exact Match": em,
            "F1": f1,
            "BERTScore": bert_f1,
            "BLEU": bleu
        })
        
        # Задержка между запросами
        time.sleep(delay)
    
    # Создаем DataFrame результатов
    results_df = pd.DataFrame(results)
    
    # Добавляем строку со средними значениями по метрикам
    avg_metrics = {
        "Вопрос": "Среднее",
        "Эталонный Ответ": "",
        "Промпт": "",
        "Ответ Модели": "",
        "Exact Match": results_df["Exact Match"].mean(),
        "F1": results_df["F1"].mean(),
        "BERTScore": results_df["BERTScore"].mean(),
        "BLEU": results_df["BLEU"].mean()
    }
    results_df = pd.concat([results_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    
    # Сохраняем результаты в Excel
    results_df.to_excel(save_file, index=False)
    print(f"\nРезультаты сохранены в {save_file}")
    
    # Выводим средние значения по метрикам в консоль
    print("\nСредние значения по метрикам:")
    print(f"Exact Match: {avg_metrics['Exact Match']:.4f}")
    print(f"F1: {avg_metrics['F1']:.4f}")
    print(f"BERTScore: {avg_metrics['BERTScore']:.4f}")
    print(f"BLEU: {avg_metrics['BLEU']:.4f}")

# Пример использования:
if __name__ == "__main__":
    from Rag import RAGSystem  # Импортируем ваш класс RAGSystem

    GROUNDED_SYSTEM_PROMPT = "Your task is to answer the user's questions using only the information from the provided documents. Give two answers to each question: one with a list of relevant document identifiers and the second with the answer to the question itself, using documents with these identifiers."

    PROMPT = PromptTemplate(
        input_variables=["context", "user_query"],
        template=(
            "Ты помощник деканата – бот, который работает с базой вопросов и ответов. "
            "Твоя задача – найти в context наиболее подходящий вопрос, соответствующий user_query, "
            "и скопировать его ответ БЕЗ изменений. Используй только один ответ, не добавляя ничего от себя.\n\n"

            "Инструкция:\n"
            "1. В context содержатся несколько вопросов и ответов, но только один из них соответствует user_query.\n"
            "2. **Найди наиболее релевантный вопрос в context и выбери ответ к нему.**\n"
            "3. **Выдай ответ пользователю без изменений.**\n"
            "4. **Если вопрос не связан с работой деканата – сообщи об этом.**\n"
            "5. **Если информации нет, попробуй уточнить запрос или предложи обратиться к оператору.**\n"
            "6. **Не используй слова 'Ответ:', 'Извлеченная информация:', 'Система сообщает:'. Просто напиши сам ответ.**\n"
            "7. **Если в ответе есть ссылки, то ставь их только в конце.**\n\n"
            
            "Вопрос: {user_query}\n\n"
            "Извлеченные документы, которые могут тебе помочь: {context}\n\n"
        )
    )

    sample_history = [{'role': 'system', 'content': GROUNDED_SYSTEM_PROMPT}, 
                      

    # Создаем экземпляр RAGSystem (с нужными параметрами)
    rag = RAGSystem(
        faiss_index_path=r"C:\Users\USER\Desktop\faiss_index",
        openai_api_base="http://10.2.3.122:382/v1",
        openai_api_key="EMPTY",
        model_name="VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF",
        agent_name="Name1",
        context_k=1,
        prompt_template=PROMPT,
    )
    
    excel_input = r"C:\Users\USER\Desktop\faiss_index\База вопросов для RAG-системы.xlsx"
    save_output = r"C:\Users\USER\Desktop\faiss_index\Оценка_промптов.xlsx"
    
    evaluate_prompts(excel_file=excel_input, save_file=save_output, rag=rag, delay=4, bert_lang="ru")

    
