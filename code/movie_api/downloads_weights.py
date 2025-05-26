import gdown

# Lien public vers le dossier Google Drive
url = 'https://drive.google.com/drive/folders/1VqINHytjzfn4pAWuCBLW8u93vcIJJzAD?usp=sharing'
url2 = 'https://drive.google.com/drive/folders/1nLICzrtrSpJK62QbDVZzwZ-E6B8Lgvaw?usp=sharing'

# Téléchargement du dossier (fichiers dans un sous-dossier local)
gdown.download_folder(url, quiet=False, use_cookies=False)
gdown.download_folder(url2, quiet=False, use_cookies=False)