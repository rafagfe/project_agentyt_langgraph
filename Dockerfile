FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y \
    ffmpeg \
    build-essential \
    python3-dev \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Create necessary directories
RUN mkdir -p /app/chroma_db /app/output

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Adjust permissions for the appuser
RUN chown -R appuser:appuser /app && \
    chmod -R 777 /app/chroma_db && \
    chmod -R 777 /app/output

# Switch to the non-root user
USER appuser

# Expose the port for Streamlit
EXPOSE 8510

# Run the application
CMD ["streamlit", "run", "app2.py", "--server.port=8510", "--server.address=0.0.0.0"]
