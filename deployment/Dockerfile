# Gunakan base image Python yang lebih ringan (Slim)
FROM python:3.9-slim

# Tetapkan direktori kerja
WORKDIR /app

# Salin file requirements.txt terlebih dahulu untuk cache layer
COPY requirements.txt /app/

# Install dependencies lebih efisien
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa file setelah dependensi terinstal (agar layer cache tidak ter-reset)
COPY . /app/

# Expose port 8000
EXPOSE 8000

# Menjalankan FastAPI dengan Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
