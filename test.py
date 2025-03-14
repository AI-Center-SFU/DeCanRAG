from Rag import RAGSystem



# Пример использования:
if __name__ == "__main__":
    # Передаем реальные значения в main, переопределяя заглушки по необходимости
    rag = RAGSystem(
        faiss_index_path=r"C:\Users\USER\Desktop\faiss_index",
        openai_api_base="https://agile.ai.sfu-kras.ru/v1",
        openai_api_key="EMPTY",
        model_name="VlSav/Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M-GGUF",
        agent_name="Мой RAG Агент",    # Меняем имя агента
        context_k=2                   # Например, берем 2 наиболее релевантных документа
    )
    print("🔹 Введите вопрос (или 'exit' для выхода):")
    
    while True:
        query = input("\n❓ Вопрос: ").strip()
        if query.lower() in ["exit", "выход", "quit"]:
            print("🚪 Выход из программы.")
            break

        prompt = rag.generate_prompt(query)
        print("\n🔹 Финальный промпт:")
        print(prompt)
        
        answer = rag.get_final_answer(query)
        print("\n🔹 Сгенерированный ответ:")
        print(answer)
