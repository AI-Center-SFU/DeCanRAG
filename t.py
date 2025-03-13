from langchain_community.chat_models import ChatOpenAI


llm = ChatOpenAI(
    openai_api_base="http://10.2.3.122:382/v1",  # Укажи свой сервер
    openai_api_key="EMPTY",  # Если не нужен ключ, можно оставить пустым
    model_name="yandexgpt"
)

def get_answer(user_query):
    response = llm.invoke(user_query)
    return response.content  # Если модель поддерживает OpenAI API

print(get_answer("Какая сегодня погода?"))