# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./

# Upgrade pip, setuptools, and wheel to the latest version
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN npm install -g prettier

# Debug: List files in the current directory
RUN ls -l

# Copy the rest of the application
COPY main.py funtion_tasks.py ./

# Expose the FastAPI port
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "main.py"]
