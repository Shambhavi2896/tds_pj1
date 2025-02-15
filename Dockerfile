# Use a smaller base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /

# Install system dependencies (including Node.js and npm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Verify installation
RUN node -v && npm -v && npx -v

# Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh | sh

# Verify uv installation
RUN uv --version && ls -lah /root/.local/bin/

# Copy dependencies file first (Docker caching optimization)
COPY requirements.txt ./

# Ensure requirements.txt exists
RUN test -f requirements.txt

# Install Python dependencies using uv
RUN uv pip install --system -r requirements.txt --no-cache-dir --verbose

# Copy the rest of the application
# COPY . .
COPY main.py prompts.py ./

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]