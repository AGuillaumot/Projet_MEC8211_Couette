import sys
import os

# Ajout automatique du dossier racine au chemin d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulations.verification_mms.mms_test import test

if __name__ == "__main__":
    test()