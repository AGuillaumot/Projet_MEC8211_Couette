import sys
import os

# Ajout automatique du dossier racine au chemin d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulations.Calcul_uinput.couette_test_uinput import test

if __name__ == "__main__":
    test()