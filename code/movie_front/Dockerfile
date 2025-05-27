FROM python:3.12-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Installer curl et unzip
#RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

# Installer les dépendances python
RUN pip install --no-cache-dir -r requirements.txt

RUN python downloads_weights.py

# Télécharger le zip depuis un lien direct (à remplacer par ton lien réel)
#RUN curl -L -o MLP-20M.zip "https://drive.google.com/file/d/176gciParY1pFkIN8R4_Yiu9XfWxG-lfI/view?usp=sharing"

# Dézipper dans /app/MLP-20M (ajuste le chemin si besoin)
#RUN unzip /app/MLP-20M.zip -d /app/

# Supprimer le zip pour alléger l'image finale
#RUN rm /app/MLP-20M.zip

# Expose le port 8000
EXPOSE 9000

# Définit la variable d'environnement Flask en production
ENV FLASK_ENV=production

# Lance l'application Flask
CMD ["python", "app.py"]
