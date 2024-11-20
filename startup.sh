#!/bin/bash

# Activer les journaux pour le démarrage
echo "Démarrage de l'application Flask avec Gunicorn"

# Assurez-vous que toutes les dépendances sont installées
pip install -r requirements.txt

# Vérification du montage du stockage Azure
if [ -d "/mnt/azureblob" ]; then
  echo "Le stockage monté est détecté."
else
  echo "Le stockage monté n'est pas disponible. Vérifiez votre configuration."
fi

# Lancer l'application avec Gunicorn
exec gunicorn --bind 0.0.0.0:8000 app:app