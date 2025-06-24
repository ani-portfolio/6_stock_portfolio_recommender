# Use an official Python runtime as a parent image
FROM python:3.11.6-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace model to avoid runtime download issues
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy all application files into the container
COPY app.py .
COPY src/ ./src/
COPY pages/ ./pages/
COPY media/ ./media/

# Expose the port that Streamlit will listen on
# Cloud Run automatically sets the PORT environment variable
ENV PORT 8080
EXPOSE $PORT


# Command to run the Streamlit application
# Updated Streamlit flags for Cloud Run compatibility
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
