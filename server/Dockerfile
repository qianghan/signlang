# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file (if you have one)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code and model
COPY server.py .
COPY asl_model.h5 .

# Expose port 5000
EXPOSE 5000

# Set environment variable for Flask
ENV FLASK_APP=server.py

# Run the server
CMD ["python", "server.py"]
