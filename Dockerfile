# Gunakan base image Python yang ringan (Alpine atau Slim lebih kecil ukurannya)
FROM python:3.11

# Set lingkungan kerja di dalam container
WORKDIR /app

# Salin file requirements.txt dan instal dependencies
COPY requirements.txt .

RUN pip install --upgrade pip  # Update pip sebelum install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi proyek ke dalam container
COPY . .

# Ekspos port untuk API
EXPOSE 8000

# Perintah untuk menjalankan FastAPI dengan Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
