# -*- coding: utf-8 -*-
"""
MEC8211 - Vérification et Validation en modélisation numérique
Date de création: 2024 - 03 - 29
Auteur: Alban GUILLAUMOT
"""
import numpy as np

class MonGradient:
    def __init__(self, mesh, face_data):
        self.mesh = mesh
        self.elements_coords = self.mesh.elements_coords()
        self.faces_coords = self.mesh.faces_coords()
        self.face_data = face_data

    
    def test(self):
        """
        Fonction appelle les fonctions qui vérifient Euler et la divergence
         
        Parameters
        ----------
        None
         
        Returns
        -------
        None
         
        """
        print("Test de la relation d'Euler ...")
        euler = self.test_euler(0)
        if euler :
            print("Le test d'Euler est positif")
        else :
            print("le test d'Euler est négatif")
            
        print("Test de divergence d'un champ constant ...")  
        divergence = self.test_divergence()
        if divergence :
            print("Le test de divergence est positif \n")
        else :
            print("le test de divergence est négatif \n")
        return
    
    def test_euler(self, h):
        """
        Fonction vérifie la condition d'Euler
         
        Parameters
        ----------
        h: Int
            Nombre de trous (ex : 1 trous quand on maille un rectangle contenant un cylindre).
         
        Returns
        -------
        boolean
            La condition est vérifiée ou non.
         
        """
        f = self.mesh.get_number_of_elements()       # Nombre de faces (nombre d'éléments quads ou triangles) [-]
        a = self.mesh.get_number_of_faces()          # Nombre d'arêtes sur le maillage [-]
        s = self.mesh.get_number_of_nodes()          # Nombre de sommets (nombre de noeuds dans le maillage) [-]
                                           
        return (f - a + s == 1 - h)
    
    def test_divergence(self):
       """
       Fonction vérifie la condition sur la divergence
        
       Parameters
       ----------
       None
        
       Returns
       -------
       boolean
           La condition est vérifiée ou non.
        
       """
       Div = np.zeros(self.mesh.get_number_of_elements())    # Initialisation d'un réel sur chaque élément
       
       v = (3, 1) # On impose uun champ constant
       
       number_of_boundary_faces = self.mesh.get_number_of_boundary_faces()
       
       for i_face in range(self.mesh.get_number_of_faces()):      
           
           elements = self.face_data[i_face]['elements']
           normale = self.face_data[i_face]['normal']
           
           delta_s = self.face_data[i_face]['Dnu']
           
           flux = np.dot(v, normale) * delta_s
           
           Div[elements[0]] += flux
           
           if i_face >= number_of_boundary_faces : Div[elements[1]] -= flux 
        
       Div_corrige = corrige_erreurs_numeriques(Div)
       return np.all(Div_corrige == 0)

    def compute_gradient(self, phi_data, bc_data):
        
        """
        Fonction qui implémente le calul du gradient d'un champ scalaire
    
        Parameters
        ----------
        phi_data : ndarray
            Champ scalaire où l'on cherche son gradient.
    
        Returns
        -------
        ndarray
            Gradient du champ scalaire dans chaque élément du maillage.
    
        """
        n_elements = self.mesh.get_number_of_elements()  # Renvoie le nombre d'éléments
        
        GRAD = np.zeros((n_elements, 2))                # Initialisation de la matrice gradients
        ATA = np.zeros((n_elements, 2, 2))              # Initialisation de la matrice ATA (membres de gauche non-inversés)
        B = np.zeros((n_elements, 2))                   # Initialisation de la matrice B (membres de droite)
        phi = phi_data.copy() 
        
        # Arrêtes frontières
        
        for i_face in range(self.mesh.get_number_of_boundary_faces()): # Boucle sur toutes les arêtes frontières
            
            bc_type, bc_value= bc_data [i_face]
            
            ALS = np.zeros((2,2))                                     # Remise à 0 de la matrice ALS    
            
            if bc_type in ('DIRICHLET', 'NEUMANN'):                                                         
                nx,ny = self.face_data[i_face]['normal']
                elements = self.face_data[i_face]['elements']
                
                xA, yA = self.faces_coords [i_face] 
                xT, yT = self.elements_coords [elements[0]]                    # Position xy du centre de l'élément T
                
                Dx = (xA - xT)           
                Dy = (yA - yT) 
                
                Dx_init = Dx.copy()
                Dy_init = Dy.copy()

                if ( bc_type == 'NEUMANN'  ) :           
                      
                    flux_normal = (Dx * nx + Dy * ny)
                    Dx = flux_normal * nx
                    Dy = flux_normal * ny
                   
                ALS [0, 0]  = Dx * Dx
                ALS [1, 0]  = Dx * Dy
                ALS [0, 1]  = Dy * Dx 
                ALS [1, 1]  = Dy * Dy
                
                ATA [elements[0]] += ALS
                
                # Calcul du membre de droite B
                if (bc_type == "DIRICHLET"):
                    
                    phiA = bc_value
                    phiT = phi[elements[0]]
                    
                    Dphi = (phiA - phiT)
                
                if ( bc_type == 'NEUMANN' ) :
                    
                    flux_normal = (Dx_init * nx + Dy_init * ny)
                    Dphi = flux_normal * bc_value
                    
                B [elements[0], 0] += Dx * Dphi
                B [elements[0], 1] += Dy * Dphi

        for i_face in range(self.mesh.get_number_of_boundary_faces(), self.mesh.get_number_of_faces()):
            ALS = np.zeros((2,2))
            elements = self.face_data[i_face]['elements']
            
            xA, yA = self.elements_coords [elements[1]]
            xP, yP = self.elements_coords [elements[0]]
    
            Dx = xA - xP
            Dy = yA - yP
            
            # Calcul du membre de gauche ATA
            ALS [0, 0]  = Dx * Dx
            ALS [1, 0]  = Dx * Dy
            ALS [0, 1]  = Dy * Dx
            ALS [1, 1]  = Dy * Dy
            
            ATA [elements[0]] += ALS
            ATA [elements[1]] += ALS
            
            # Calcul du membre de droite B
            phiA = phi[elements[1]]
            phiP = phi[elements[0]]
            Dphi = phiA - phiP
            
            B [elements[0], 0] +=  Dx * Dphi
            B [elements[0], 1] +=  Dy * Dphi
            B [elements[1], 0] +=  Dx * Dphi
            B [elements[1], 1] +=  Dy * Dphi
            
        # Calcul de la matrice inverse ATAI puis le gradient dans chaque élément
         
  
        for i_element in range(n_elements):  
            GRAD[i_element] = np.linalg.solve(ATA[i_element]+np.eye(2)*1e-16, B[i_element])
       
       
        return GRAD
def corrige_erreurs_numeriques(Tab, tolerance=1e-10):
    """
    Corrige l'erreur numérique pour les valeurs proches de zéro.

    Parameters
    ----------
    Tab: ndarray
        Tableau à vérifier.
    Returns
    -------
    ndarray
        Tableau corrigé.

    """
    return np.where(np.abs(Tab) < tolerance, 0, Tab)