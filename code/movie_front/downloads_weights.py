import gdown
import zipfile

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

"""
import gdown

# Lien public vers le dossier Google Drive
url = 'https://drive.google.com/file/d/176gciParY1pFkIN8R4_Yiu9XfWxG-lfI/view?usp=sharing'

# Téléchargement du dossier (fichiers dans un sous-dossier local)
gdown.download_folder(url, quiet=False, use_cookies=False)

"""