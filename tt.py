import re
import pandas as pd
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import warnings
warnings.filterwarnings("ignore")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

EXCEL_FILE = r"C:\Users\USER\Desktop\faiss_index\База вопросов для RAG-системы.xlsx"

sheets = pd.read_excel(EXCEL_FILE, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)

df["Вопрос"] = df["Вопрос"].apply(clean_text)
df["Ответ"] = df["Ответ"].apply(clean_text)
# Добавляем ссылку из столбца "Ссылка на документы" в начало столбца "Ответ"
df["Ответ"] = "Подробнее: " + df["Ссылка на документы"].fillna("") + "\n" + df["Ответ"]

loader = DataFrameLoader(df, page_content_column='Вопрос')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

documents = [
    Document(
        page_content=row["Вопрос"],
        metadata={"Ответ": row["Ответ"]}
    )
    for _, row in df.iterrows()
]

texts = text_splitter.split_documents(documents)

embedding_model = "ai-forever/sbert_large_nlu_ru"
hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

db = FAISS.from_documents(texts, hf_embeddings)

db.save_local("faiss_index")
