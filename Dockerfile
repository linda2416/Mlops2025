# # Utiliser une image de base officielle Python
# FROM python:3.9-slim

# # Définir le répertoire de travail
# WORKDIR /app

# # Copier les fichiers du projet dans l'image Docker
# COPY . /app

# # Installer les dépendances
# RUN pip install --no-cache-dir -r requirements.txt

# # Exposer le port sur lequel l'application s'exécute
# EXPOSE 8000

# # Démarrer le service FastAPI
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]




# Use the official Jenkins image as the base
FROM jenkins/jenkins:lts

# Switch to root user to install Python
USER root

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Switch back to the Jenkins user
USER jenkins

# Set the working directory
WORKDIR /app

# Copy project files into the Docker image
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port on which the application runs
EXPOSE 8000

# Start the FastAPI service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
