FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Make sure the models directory exists
RUN mkdir -p models

# Set the necessary environment variables
ENV FLASK_APP=app/app.py
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose the port the app runs on
EXPOSE 5000

# Command to run the app
CMD ["python", "app/app.py"]