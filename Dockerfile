# Dockerfile

# Start from an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code (engine.py and app.py) into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 5000

# Define the command to run your app when the container starts
# Gunicorn is a production-ready web server
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "app:app"]