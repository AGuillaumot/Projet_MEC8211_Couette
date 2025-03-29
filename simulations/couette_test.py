# -*- coding: utf-8 -*-
"""
MEC8211 - Vérification et Validation en modélisation numérique
Date de création: 2024 - 03 - 29
Auteur: Alban GUILLAUMOT
"""
from src import Discretisation_Volumes_Finis, init_face_data, MeshGenerator, MeshConnectivity, MonPlotter
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import time

def test():
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
    mu = 1               # Viscosité dynamique   [Ns/m^2]
    n_iter = 2000        # Nombre d'itération de l'algorithme simple [-]
    P = 2        # Pression
    
    coeff_data = np.array([rho, mu, n_iter])
    
    geometry = [0, 5, 0, 1] # Dimension de notre surface de contrôle
    
    #---------------------- Conditions aux limites  ------------------------------------- 
    # Implémentation de toutes les fonctions 
    
    y = sp.symbols('y')   
    unitaire_function = sp.lambdify(y,1)    # Fonction unitaire pour une entrée avec profil plat
    zero_function = sp.lambdify(y,0)        # Fonction nulle
    partial_function =  sp.lambdify(y,(1+P)/2-P/3)
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -      
    #  Conditions aux limites sur les vitesses, on impose :
    #   - DIRICHLET en entrée avec u = constante et v = 0
    #   - DIRICHLET sur la paroi inférieure avec u = v = 0 m/s  
    #   - NEUMANN en sortie avec grad_u = grad_v = [0,0], on considère que la normale est horizontale et que la deuxième composante n'intervient pas dans les calculs
    #   - DIRICHLET sur la paroi supérieure avec u = 1 et v = 0 m/s
    
    bc_speed_data = ([['DIRICHLET','DIRICHLET'], [partial_function, zero_function], [None, None]],
                     [['DIRICHLET','DIRICHLET'], [zero_function, zero_function], [None, None]],
                     [['NEUMANN','NEUMANN'], [None, None], [[0,0], [0,0]]],
                     [['DIRICHLET','DIRICHLET'], [unitaire_function, zero_function], [None, None]])
    
    #  Conditions aux limites sur les pressions, on impose :
    #   - NEUMANN en entrée avec grad_p = [-2P,0] 
    #   - LIBRE sur la paroi inférieure (aucune contribution)
    #   - DIRICHLET en sortie avec p = 0
    #   - LIBRE sur la paroi supérieure (aucune contribution)
    
    bc_pressure_data = ( [['NEUMANN', None, [0,-2*P]],['LIBRE', None, None],
                 ['LIBRE', None, [0,0]],['DIRICHLET', zero_function, None]])
    
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
    
    print("Maillage rectangulaire transfini (nx=10,ny=5), éléments QUAD et schéma Upwind \n")
    
    print("Initialisation du maillage...")
    mesh_parameters_P1C1 = {'mesh_type': 'QUAD','Nx': 10,'Ny': 5}
    
    mesh_obj_P1C1 = mesher.rectangle(geometry, mesh_parameters_P1C1)
    scheme_P1C1 = "UPWIND"
    print("La simulation 1 comporte {} éléments !".format(mesh_obj_P1C1.get_number_of_elements()))
    
    conec_P1C1 = MeshConnectivity(mesh_obj_P1C1, verbose= False)
    conec_P1C1.compute_connectivity() 
    face_data1, bc_u1, bc_v1, bc_p1 = init_face_data(mesh_obj_P1C1, bc_speed_data, bc_pressure_data)    # Initialise tous les coefficients liés aux faces de notre maillage et aux conditions aux limites
    
    print("\nDébut des itérations de l'algorithme SIMPLE")
    P1C1 = Discretisation_Volumes_Finis(mesh_obj_P1C1, face_data1, bc_u1, bc_v1, bc_p1, bc_door_data, coeff_data, scheme_P1C1, geometry)
    u1, v1, p1, grad_p1, b1 = P1C1.algorithme_simple()  
    
    Plotter1 = MonPlotter(mesh_obj_P1C1, geometry)
    print("\nVoir sur Pyvista les champs de vitesse et de pression")
    Plotter1.my_plotter_2D(u1, "Champ de vitesse u suivant x pour le cas 1")
    Plotter1.my_plotter_2D(p1, "Champ de pression pour le cas 1")
    Plotter1.my_plotter_3D(u1, v1, "Champ de vitesse pour le cas 1")
    
    
    print("Voir sur le module _Graphes_ les comparaisons du profil de vitesse axial ainsi que du gradient de pression axial à l’approche de la sortie avec la solution analytique pour le cas pleinement développé")
    points_u1 = Plotter1.my_plotter_1D(u1, False)
    plt.plot(points_u1[:,1], points_u1[:,0], 'g', label='Simulation u', marker='o')
    plt.plot(np.linspace(0,1,50)*(1+P*(1-np.linspace(0,1,50))), np.linspace(0,1,50), 'r', label='Théorie u')
    plt.legend()
    plt.xlabel("Vitesse axiale [m/s]")
    plt.ylabel("Position Y [m]")
    plt.title("Comparaison du profil de vitesse axial à l’approche de la sortie avec la solution analytique pour le cas 1")
    plt.show()
    
    print("Voir sur le module _Graphes_ la convergence itérative de l’algorithme SIMPLE. \n")
    plt.plot(b1)
    plt.yscale('log')
    plt.xlabel("Iteration [-]")
    plt.ylabel("Divergence [-]")
    plt.title("Convergence itérative de l’algorithme SIMPLE pour le cas 1")
    plt.show()
    
    log_time("réaliser le cas 1")

    return 

