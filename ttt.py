import re
import pandas as pd

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Путь к исходному файлу и файл для сохранения обновлённой базы
EXCEL_FILE = r"C:\Users\USER\Desktop\faiss_index\База вопросов для RAG-системы.xlsx"
OUTPUT_FILE = r"C:\Users\USER\Desktop\faiss_index\База вопросов для RAG-системы_обновленный.xlsx"

# Читаем все листы и объединяем их в один DataFrame
sheets = pd.read_excel(EXCEL_FILE, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)

# Очистка столбцов "Вопрос" и "Ответ"
df["Вопрос"] = df["Вопрос"].apply(clean_text)
df["Ответ"] = df["Ответ"].apply(clean_text)

# Добавляем ссылку из столбца "Ссылка на документы" в конец столбца "Ответ"
df["Ответ"] = df["Ответ"] + "\nПодробнее: " + df["Ссылка на документы"].fillna("")

# Сохраняем обновлённую таблицу в новый Excel-файл
df.to_excel(OUTPUT_FILE, index=False)
