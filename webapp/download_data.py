import gdown
import zipfile
import os

# ID du fichier (extraite depuis l'URL Google Drive)
file_id = "176gciParY1pFkIN8R4_Yiu9XfWxG-lfI"

# Lien direct compatible avec gdown
url = f"https://drive.google.com/uc?id={file_id}"

# Nom du fichier de sortie
output = "MLP-20M.zip"

# Téléchargement
gdown.download(url, output, quiet=False)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall()

# Suppression du fichier zip après extraction
os.remove(output)