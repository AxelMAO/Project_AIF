FROM python:3.12-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Installer curl et unzip
#RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

# Installer les dépendances python
RUN pip install --no-cache-dir -r requirements.txt

RUN python download_data.py

# Expose le port 8000
EXPOSE 8000

# Définit la variable d'environnement Flask en production
ENV FLASK_ENV=production

# Lance l'application Flask
CMD ["python", "app.py"]
