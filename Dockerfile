FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN mkdir -p model_weights saved

COPY app.py .
COPY model_weights/model_119.keras model_weights/model_119.keras
COPY saved/word_to_idx.pkl saved/word_to_idx.pkl
COPY saved/idx_to_word.pkl saved/idx_to_word.pkl
COPY saved/max_len.pkl saved/max_len.pkl
RUN ls -lh model_weights/ saved/
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]