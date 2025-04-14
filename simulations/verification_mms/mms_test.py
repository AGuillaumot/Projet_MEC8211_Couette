# -*- coding: utf-8 -*-
"""
MEC8211 - Vérification et Validation en modélisation numérique
Date de création: 2024 - 03 - 29
Auteur: Alban GUILLAUMOT
"""
from src import MeshConnectivity, MeshGenerator, MonPlotter

from simulations.verification_mms.mms_core import Discretisation_Volumes_Finis, init_face_data
from simulations.verification_mms.mms_definitions import u_MMS, v_MMS, p_MMS, source_mms

import numpy as np
import matplotlib.pyplot as plt
import time
import os


def test():
    # Création d’un nouveau dossier run automatiquement (placé ICI dans test)
    base_output = "results/resultats_1_MMS"
    os.makedirs(base_output, exist_ok=True)

    # Générer un ID de run unique
    run_id = 1
    while os.path.exists(os.path.join(base_output, f"run_{run_id:03d}")):
        run_id += 1

    # Créer le dossier du run courant
    output_folder = os.path.join(base_output, f"run_{run_id:03d}")
    os.makedirs(output_folder)

    # Créer le sous-dossier pour les graphiques
    graph_folder = os.path.join(output_folder, "graphs")
    os.makedirs(graph_folder)
    """
    Fonction pour tester le calcul du champ de vitesse.

    Parameters
    ----------
    None

    Returns
    -------
    None
    
   """
    #---------------------- Gestion affichage temps ---------------------- 
    def log_time(label): # Fonction pour afficher le temps d'une étape
        nonlocal last_time
        current_time = time.time()
        print(f"Temps pour {label} : {round(current_time - last_time, 2)}s")
        last_time = current_time  # Mettre à jour le dernier temps
        
    #---------------------- PROBLÈME 1 : ECOULEMENT DE COUETTE  ---------------------- 
    
    print("PROBLÈME 1 : Ecoulement de Couette \n")
    
    #---------------------- Paramètres du problème  ------------------------------------- 
                  
    rho = 1                 # Masse volumique       [kg/m^3]
    mu =  0.1               # Viscosité dynamique   [Ns/m^2]
    n_iter = 4000        # Nombre d'itération de l'algorithme simple [-]
 # --- TEST COHÉRENCE SOURCES MMS ---
    print("\n🔍 Vérification des sources MMS à quelques points...")

    points_test = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    for x, y in points_test:
        fx, fy = source_mms(x, y, mu, rho)
        print(f"  À (x={x:.2f}, y={y:.2f}) : fx = {fx:.3e}, fy = {fy:.3e}")    
    coeff_data = np.array([rho, mu, n_iter])
    
    n_raff = 5
    r = 1.3
    Nx_min = 12
    Ny_min = 12
    
    geometry = [0, 1, 0, 1] # Dimension de notre surface de contrôle
    
    #---------------------- Conditions aux limites  -------------------------------------      
    bc_speed_data = ([['DIRICHLET','DIRICHLET'], [u_MMS, v_MMS], [None, None]],
                     [['DIRICHLET','DIRICHLET'], [u_MMS, v_MMS], [None, None]],
                     [['DIRICHLET','DIRICHLET'], [u_MMS, v_MMS], [None, None]],
                     [['DIRICHLET','DIRICHLET'], [u_MMS, v_MMS], [None, None]])
    
    bc_pressure_data = (
        ['DIRICHLET', p_MMS, None],         # INLET (x = 0)
        ['NEUMANN', None, [0, 0]],          # WALL bas (y = 0)
        ['NEUMANN', None, [0, 0]],          # OUTLET (x = 5)
        ['NEUMANN', None, [0, 0]]           # WALL haut (y = 1)
)
    
    #  Conditions aux limites sur les contours de notre surface de contrôle, on impose :
    #   - INLET à gauche (TAG=0 et x=0) pour une entrée
    #   - WALL en bas (TAG=1 et y=0) pour une paroi
    #   - OUTLET à droite (TAG=2 et x=5) pour une sortie
    #   - WALL en haut (TAG=3 et y=1) pour une paroi
    bc_door_data = ['INLET', 'WALL', 'OUTLET','WALL']
    
        
    #---------------------- Maillage et Schéma  ---------------------------------------------------------
    mesher = MeshGenerator()
    
    #---------------------- Rectangle transfini (nx=10,ny=5), QUAD, Upwind ---------------------- 
    start = time.time()
    last_time = start  # Pour stocker le temps de la dernière étape
    
    h = np.zeros(n_raff)
    u_norm_UPWIND = np.zeros((n_raff, 3))
    v_norm_UPWIND = np.zeros((n_raff, 3))
    
    u_norm_CENTRE = np.zeros((n_raff, 3))
    v_norm_CENTRE = np.zeros((n_raff, 3))
    
    for i_raff in range(n_raff):
        Nx = round(Nx_min * r**(i_raff))
        Ny = round(Ny_min * r**(i_raff))
        
        Lx = geometry[1] - geometry[0]  # 5
        Ly = geometry[3] - geometry[2]  # 1
        hx = (Lx / Nx)**0.5
        hy = (Ly / Ny)**0.5
        h[i_raff] = max(hx, hy)
        
        print("Maillage rectangulaire transfini (nx={},ny={}), éléments QUAD et schéma Upwind \n".format(Nx,Ny))
        
        print("Initialisation du maillage...")
        mesh_parameters = {'mesh_type': 'QUAD','Nx': Nx,'Ny': Ny}
        
        mesh_obj = mesher.rectangle(geometry, mesh_parameters)
        scheme_UPWIND = "UPWIND"
        scheme_CENTRE = "CENTRE"
        
        n_elements = mesh_obj.get_number_of_elements()
        u_mms = np.zeros(n_elements)
        v_mms = np.zeros(n_elements)
        p_mms = np.zeros(n_elements)
        element_coords = mesh_obj.elements_coords()
        for i_element in range(n_elements):
            xE, yE = element_coords[i_element]
            u_mms[i_element] = u_MMS(xE, yE)
            v_mms[i_element] = v_MMS(xE, yE)
            p_mms[i_element] = p_MMS(xE, yE)
        
        print("La simulation {} comporte {} éléments !".format(i_raff+1, n_elements))
        
        conec = MeshConnectivity(mesh_obj, verbose= False)
        conec.compute_connectivity() 
        face_data, bc_u, bc_v, bc_p = init_face_data(mesh_obj, bc_speed_data, bc_pressure_data)    # Initialise tous les coefficients liés aux faces de notre maillage et aux conditions aux limites
        
        UPWIND = Discretisation_Volumes_Finis(mesh_obj, face_data, bc_u, bc_v, bc_p, bc_door_data, coeff_data, scheme_UPWIND, geometry)
        CENTRE = Discretisation_Volumes_Finis(mesh_obj, face_data, bc_u, bc_v, bc_p, bc_door_data, coeff_data, scheme_CENTRE, geometry)
        
        
        print("\nDébut des itérations de l'algorithme SIMPLE UPWIND")
        u_UPWIND, v_UPWIND, p_UPWIND, grad_p_UPWIND, b_UPWIND = UPWIND.algorithme_simple()  
        
        print("\nDébut des itérations de l'algorithme SIMPLE CENTRE \n")
        u_CENTRE, v_CENTRE, p_CENTRE, grad_p_CENTRE, b_CENTRE = CENTRE.algorithme_simple()
        
        u_norm_UPWIND [i_raff] = erreur_norm(u_UPWIND, u_mms)
        v_norm_UPWIND [i_raff] = erreur_norm(v_UPWIND, v_mms)
        
        u_norm_CENTRE [i_raff] = erreur_norm(u_CENTRE, u_mms)
        v_norm_CENTRE [i_raff] = erreur_norm(v_CENTRE, v_mms)
        
        log_time("réaliser le cas 1")
    
    erreur_print(h, u_norm_CENTRE, u_norm_UPWIND, "u", graph_folder)
    erreur_print(h, v_norm_CENTRE, v_norm_UPWIND, "v", graph_folder)
    log_path = os.path.join(output_folder, "resultats_console.txt")
    with open(log_path, "w") as f:
        def print_and_log(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=f)
    
        print_and_log("\nOrdre de convergence CENTRE")
        print_and_log("\nPour u :")
        ordre = erreur_ordre(h, u_norm_CENTRE, 'CENTRE u', n_raff)
        for i, label in enumerate(['1', '2', 'inf']):
            print_and_log(f"Ordre {label} : {ordre[i]}")
    
        print_and_log("\nPour v :")
        ordre = erreur_ordre(h, v_norm_CENTRE, 'CENTRE v', n_raff)
        for i, label in enumerate(['1', '2', 'inf']):
            print_and_log(f"Ordre {label} : {ordre[i]}")
    
        print_and_log("\nOrdre de convergence UPWIND")
        print_and_log("\nPour u :")
        ordre = erreur_ordre(h, u_norm_UPWIND, 'UPWIND u', n_raff)
        for i, label in enumerate(['1', '2', 'inf']):
            print_and_log(f"Ordre {label} : {ordre[i]}")
    
        print_and_log("\nPour v :")
        ordre = erreur_ordre(h, v_norm_UPWIND, 'UPWIND v', n_raff)
        for i, label in enumerate(['1', '2', 'inf']):
            print_and_log(f"Ordre {label} : {ordre[i]}")
    
    return
def erreur_ordre (h, norm, titre, n_raff):
    
    ordre = np.zeros(3)
    for i in range (3):
        ordre[i] = np.log(norm[n_raff-2, i]/norm[n_raff-1, i]) / np.log(h[n_raff-2]/h[n_raff-1])
        
    print("\n La valeur de l'ordre de convergence calculé avec la norme 1 {} est : {}".format(titre, ordre[0]))
    print("\n La valeur de l'ordre de convergence calculé avec la norme 2 {} est : {}".format(titre, ordre[1]))
    print("\n La valeur de l'ordre de convergence calculé avec la norme infinie {} est : {}".format(titre, ordre[2]))
    
    return ordre

def erreur_print(h, CENTRE, UPWIND, val, graph_folder):
    plt.figure()

    # Courbes pour CENTRE
    plt.plot(h, CENTRE[:,0], 'g', label='Norme inf CENTRE', linestyle='-.', marker='o')
    plt.plot(h, CENTRE[:,1], 'r', label='Norme 1 CENTRE', linestyle='--', marker='o')
    plt.plot(h, CENTRE[:,2], 'b', label='Norme 2 CENTRE', marker='o')

    # Courbes pour UPWIND
    plt.plot(h, UPWIND[:,0], 'c', label='Norme inf UPWIND',linestyle='-.',)
    plt.plot(h, UPWIND[:,1], 'm', label='Norme 1 UPWIND', linestyle='--')
    plt.plot(h, UPWIND[:,2], 'y', label='Norme 2 UPWIND')

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel("Pas de maillage h (log)")
    plt.ylabel(f"Erreur sur {val}")
    plt.title(f"Évolution de la norme de l'erreur MMS en échelle log-log ({val})")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_folder, f"erreur_{val}.png"))
    plt.show()  # ← ici pour Spyder
    plt.close()



def erreur_norm (theorie, analytique):
    n_elements = len(theorie)

    n_1 = np.sum((np.abs(theorie - analytique))) / n_elements
    n_2 = np.sqrt(np.sum(((theorie - analytique)**2)) / n_elements) 
    n_inf = np.max(np.abs(theorie - analytique))
    
    return np.array([n_1, n_2, n_inf]) 

if __name__ == "__main__":
    test()