FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY reports/example-input.json ./reports/example-input.json
RUN pip install --no-cache-dir --no-deps .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
