import re


def clean_text(text):
    # Убираем лишние обратные слеши и пробелы
    text = text.replace("\\n", " ")

    # Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text)

    # Убираем пробелы перед точками и запятыми
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)

    return text.strip()
