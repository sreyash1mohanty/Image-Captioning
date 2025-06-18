FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN mkdir -p model_weights saved

COPY app.py .
COPY model_weights/ model_weights/
COPY saved/ saved/

RUN echo "Validating model file..." && \
    if [ ! -f "model_weights/model_119.keras" ]; then \
        echo "Model file not found!" && exit 1; \
    fi && \
    file_size=$(du -h "model_weights/model_119.keras" | cut -f1) && \
    echo "Model file size: $file_size" && \
    python -c "import os; size=os.path.getsize('model_weights/model_119.keras'); exit(1) if size < 1000000 else print('✅ Model size OK')" || { echo "❌ Model file too small!"; exit 1; }

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]