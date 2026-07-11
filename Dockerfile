FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY model ./model
COPY model_helper.py .
COPY app.py .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]