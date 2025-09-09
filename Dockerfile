
FROM python:3.11-slim-bookworm

# --- Variables de entorno de ejecución ---
# --- Runtime env ---
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# --- Paquetes del sistema ---
# Install only what we need (no recommends) and keep image small
# Also apply security updates via `apt-get upgrade -y` to pick up patched packages
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Dependencias de Python ---
# Leverage Docker cache: install deps first
COPY requirements.txt ./
RUN pip install --upgrade pip wheel setuptools \
 && pip install --no-cache-dir -r requirements.txt

# --- Copiar el código de la aplicación ---
# Copy application code (include subfolders like dashboard/, utils/, etc.)
COPY . .

# --- Crear usuario no root ---
# Create a non-root user for better security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# --- Exponer el puerto de la aplicación ---
EXPOSE 8501

# --- Comprobación de estado del contenedor ---
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# --- Comando de entrada del contenedor ---
ENTRYPOINT ["streamlit", "run", "app_core_act.py", "--server.port=8501", "--server.address=0.0.0.0"]