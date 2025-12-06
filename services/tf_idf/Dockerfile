FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .

# Étape 2 : installer les dépendances (Docker les garde en cache)
RUN pip install --no-cache-dir -r requirements.txt

# Étape 3 : copier le reste du code
COPY . .
EXPOSE 8001
CMD ["uvicorn", "tf-idf_service:app", "--host", "0.0.0.0", "--port", "8001"]
