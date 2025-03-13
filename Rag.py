from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import VLLMOpenAI
import warnings
warnings.filterwarnings("ignore")
from langchain.prompts import PromptTemplate

# Определяем дефолтный шаблон промпта
DEFAULT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "user_query"],
    template=(
        "Используй следующую информацию для ответа:\n\n"
        "{context}\n\n"
        "Вопрос: {user_query}\n"
        "Ответ:"
    )
)

class RAGSystem:
    def __init__(self,
                 embedding_model: str = "ai-forever/sbert_large_nlu_ru",  
                 faiss_index_path: str = "faiss_index",                    
                 openai_api_base: str = "http://localhost:8000/v1",         
                 openai_api_key: str = "EMPTY",                             
                 model_name: str = "yandex/YandexGPT-5-Lite-8B-pretrain",
                 prompt_template: PromptTemplate = DEFAULT_PROMPT_TEMPLATE,
                 agent_name: str = "RAG Agent",         # Имя RAG-агента
                 context_k: int = 1                     # Число документов для поиска релевантного контекста
                ):
        # Сохраняем имя агента
        self.agent_name = agent_name
        
        # Сохраняем параметр количества документов для контекста
        self.context_k = context_k
        
        # Инициализация эмбеддингов
        self.hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        # Загрузка FAISS-индекса
        self.vector_db = FAISS.load_local(faiss_index_path, self.hf_embeddings,
                                          allow_dangerous_deserialization=True)
        # Инициализация модели для генерации ответа
        self.llm = VLLMOpenAI(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            best_of=3,
            temperature=0.3,

        )
        # Сохраняем шаблон как атрибут
        self.prompt_template = prompt_template

    def get_context(self, query: str) -> str:
        """
        Ищет в базе знаний self.context_k наиболее релевантных документов.
        Извлекает из каждого документа пару "Вопрос - Ответ" и возвращает объединённый текст.
        Вопрос берётся из основного текста документа, ответ – из метаданных.
        """
        relevants = self.vector_db.similarity_search(query, k=self.context_k)
        if not relevants:
            return "Информация не найдена."
        context = "\n\n".join([
            f"Вопрос: {doc.page_content}\nОтвет: {doc.metadata.get('Ответ', 'Ответ не найден')}"
            for doc in relevants
        ])
        return context

    def generate_prompt(self, user_query: str) -> str:
        """
        Формирует промпт для модели, объединяя найденный контекст и вопрос пользователя,
        с использованием заранее определенного шаблона.
        """
        context = self.get_context(user_query)
        prompt = self.prompt_template.format(context=context, user_query=user_query)
        return prompt

    def get_final_answer(self, user_query: str) -> str:
        """
        Генерирует финальный ответ от модели на основе сформированного промпта.
        """
        prompt = self.generate_prompt(user_query)
        response = self.llm.invoke(prompt)
        return response


