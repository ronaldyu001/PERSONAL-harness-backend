# ___________________________________ 
#   Builder Phase. 
# ___________________________________
# Choose a base image.
FROM python:3.12-slim as builder

# Set the working directory.
WORKDIR /usr/app

# Skip .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Print logs immediately.
ENV PYTHONUNBUFFERED=1

# Create a venv and prepend its path to PATH.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requiremets.txt, Upgrade pip, and Install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ___________________________________ 
#   Runtime Phase.
# ___________________________________
FROM python:3.12-slim AS runtime

WORKDIR /usr/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
COPY . .

# Add a non root user, and disable interactivity. Switch to non root user.
RUN adduser \
    --disabled-password \
    --gecos "" \
    appuser && \
    mkdir -p /home/appuser/.mem0 && \
    chown -R appuser:appuser /home/appuser/.mem0
USER appuser

EXPOSE 8000
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
