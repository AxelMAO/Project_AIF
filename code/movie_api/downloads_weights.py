import gdown

# Lien public vers le dossier Google Drive
url = 'https://drive.google.com/drive/folders/1VqINHytjzfn4pAWuCBLW8u93vcIJJzAD?usp=sharing'

# Téléchargement du dossier (fichiers dans un sous-dossier local)
gdown.download_folder(url, quiet=False, use_cookies=False)
