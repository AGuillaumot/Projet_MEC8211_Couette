from src.monGradient import MonGradient
from simulations.verification_mms.mms_definitions import source_mms

import numpy as np
from scipy.sparse import lil_matrix,csr_matrix
from scipy.sparse.linalg import spsolve
  
class Discretisation_Volumes_Finis ():
    def __init__(self, mesh, face_data, bc_u, bc_v, bc_p, bc_door_data, coeff_data, scheme, geometry):
        self.mesh = mesh
        self.elements_area = self.mesh.elements_area()
        self.Gradient = MonGradient(mesh, face_data)
        self.bc_u, self.bc_v, self.bc_p = bc_u, bc_v, bc_p
        self.bc_door_data = bc_door_data
        self.coeff_data = coeff_data
        self.scheme = scheme
        self.face_data = face_data
           
    def algorithme_simple(self):
        """
            Résout un problème de dynamique des fluides en utilisant l'algorithme SIMPLE.
            
            Parameters
            ----------
            None
                
            Returns
            -------
            ndarray
                champ de vitesse dans la direction x.
            ndarray 
                champ de vitesse dans la direction y.
            ndarray
                champ de pression.
            ndarray
                gradient de pression.
            list
                divergence à chaque itération.
        """

        # 1 : Estimer un champ de vitesses aux centres des éléments u0 et v0 et de pression p0 
        n_elements = self.mesh.get_number_of_elements()
        u = np.zeros(n_elements)                        # Initialisation de u à 0
        v = np.zeros(n_elements)                        # Initialisation de v à 0
        pressure = np.zeros(n_elements)                 # Initialisation de p à 0
        grad_pressure = np.zeros((n_elements,2))        # Initialisation de grad(p) à [0,0]
        
        # 2 : Initialiser les flux débitants aux faces F0
        flow = self.calcul_flow_init(u, v)              # Initialisation des flux débitants à partir des conditions aux limites et de u,v
        
        n_iter = int(self.coeff_data[2])                
        b = []                                          # Initialisation de la liste des divergences
        i_iter = 0                                      # Indice des itérations de l'algorithme SIMPLE
        div = 1
        
        source_MMS = np.zeros((n_elements, 2))
        elements_coords = self.mesh.elements_coords()
        
        rho = self.coeff_data[0]
        mu = self.coeff_data[1]
        
        for i_element in range(n_elements):
            xT, yT = elements_coords[i_element]    
            fx, fy = source_mms(xT, yT, mu, rho)
            source_MMS[i_element, 0] = fx
            source_MMS[i_element, 1] = fy
        while i_iter < n_iter and div > 1e-5 :          # On veut arrêter la boucle si la divergence est inférieure à 1e-5 ou si on a atteint le nombre d'itération maximum
            
            # 3 : Résolution des équations du mouvement pour obtenir u_m et v_m
            grad_pressure = self.Gradient.compute_gradient(pressure, self.bc_p)     # Calcul du gradient de pression à partir de la pression précédente
            source = np.zeros((n_elements, 2))                          
            
            for i_element in range(n_elements):                                     # Calcul du terme source dans chaque élément
                source [i_element] = -self.elements_area[i_element] * grad_pressure[i_element]
                source [i_element, 0] += source_MMS[i_element, 0]* self.elements_area[i_element]
                source [i_element, 1] += source_MMS[i_element, 1]* self.elements_area[i_element]
            u, v, A = self.calcul_momentum(flow, source)                            # Calcul du momentum avec les flux débitants et le terme source
            
            # 4 : Calcul des flux aux faces f_RC par la méthode de Rhie et Chow.
            f_RC = self.fonction_RhieChow(u, v, A, pressure, grad_pressure)         # Calcul des flux aux faces avec Rhie Chow
            
            alpha_rc = 0.1          
            f_sr = alpha_rc * f_RC + (1-alpha_rc)*flow                              # On sous-relaxe les flux aux faces obtenus précédement
           
            # 5 et 6 : Résoudre l’équation de correction de pression, obtenir les P’ et corriger les flux aux faces F** pour obtenir les Fn+1 en utilisant les P’
            flow, p_corrige, div = self.correction_pression(A, f_sr)                # Corrige les flux débitants et la pression et donne la valeur de la divergence
            b.append(div)                                                           # Ajoute la nouvelle valeur de divergence à la liste des divergences
            
            #7. Correction du champ de pression avec sous-relaxation
            alpha_p = 0.1
            pressure += alpha_p * p_corrige                                         # On sous-relaxe les flux aux faces obtenus précédement                                     
            if i_iter % 250 ==0:
                print("Itération : {} et b : {}".format(i_iter,np.format_float_scientific(b[i_iter], precision=2)))
            i_iter+=1
            
        print("Le programme s'est arrêté à l'itération n° {} pour une divergence de {}".format(i_iter-1,np.format_float_scientific(b[i_iter-1], precision=4)))   
        return u, v, pressure, grad_pressure, b
    
    def calcul_flow_init(self, u, v):
        """
        Calcule les flux initiaux aux faces d'un maillage en tenant compte des conditions aux limites.
    
        Parameters
        ----------
        u : ndarray
            Champ de vitesse dans la direction x pour chaque élément du maillage.
        v : ndarray
            Champ de vitesse dans la direction y pour chaque élément du maillage.
    
        Returns
        -------
        flow_init : ndarray
            Tableau contenant les flux initiaux calculés pour chaque face du maillage.
        """

        n_faces = self.mesh.get_number_of_faces()
       
        flow_init = np.zeros(n_faces)
        
        for i_face in range(self.mesh.get_number_of_boundary_faces()): #  On prend en compte les conditions aux limites pour les faces externes
            i_face_data = self.face_data[i_face]
                         
            bc_type_u , bc_value_u = self.bc_u [i_face]
            bc_type_v , bc_value_v = self.bc_v [i_face]
            
            Fxi = 0
            Fyi = 0
            
            if (bc_type_u == 'NEUMANN'):

                Fxi = u[i_face_data['elements'][0]] + bc_value_u * i_face_data['Dxi'] * i_face_data['PNKSI']
                                    
            elif (bc_type_u == 'DIRICHLET'):

                Fxi = bc_value_u
                
            if (bc_type_v == 'NEUMANN'):
                
                Fyi = v[i_face_data['elements'][0]] + bc_value_v * i_face_data['Dxi'] * i_face_data['PNKSI']
                
            elif (bc_type_v == 'DIRICHLET'):
  
                Fyi = bc_value_v

            flow_init [i_face] = np.dot(i_face_data['normal'], [Fxi, Fyi])
            
            
        for i_face in range(self.mesh.get_number_of_boundary_faces(), n_faces): # On fait une moyenne simple des vitesses connues pour les faces internes
            i_face_data = self.face_data[i_face]
            elements = i_face_data['elements']
      
            phi_x = (u[elements[0]]+u[elements[1]])/2
            phi_y = (v[elements[0]]+v[elements[1]])/2
            
            flow_init [i_face] = np.dot([phi_x,phi_y], i_face_data['normal'])
        return flow_init
    
    def calcul_momentum(self, f, S):
        """
        Calcule les champs de vitesse u et v en résolvant les équations de la quantité de mouvement, 
        en tenant compte de la cross-diffusion et des conditions aux limites.
    
        Parameters
        ----------
        f : ndarray
            Flux débitant donné pour chaque face du maillage.
        S : ndarray
            Terme source.
    
        Returns
        -------
        
        u : ndarray
            champ de vitesse corrigé dans la direction x.
        v : ndarray 
            champ de vitesse corrigé dans la direction y.
        A_x : csr_matrix
            matrice des coefficients pour le calcul de u.
        """
             
        n_elements = self.mesh.get_number_of_elements()     # Renvoie le nombre d'éléments
        n_faces = self.mesh.get_number_of_faces()           # Renvoie le nombre de faces
        
        rho = self.coeff_data[0]
        mu = self.coeff_data[1]

        u = np.zeros(n_elements)                            # Initialisation de u à 0
        v = np.zeros(n_elements)                            # Initialisation de v à 0

        A_x = lil_matrix((n_elements,n_elements))      
        A_y = lil_matrix((n_elements,n_elements))
        
        B_x = S[:,0].copy()                             # On initialise le membre de droite au terme source
        B_y = S[:,1].copy()                             # On initialise le membre de droite au terme source
              
        for i_face in range(self.mesh.get_number_of_boundary_faces()):
            
            i_face_data = self.face_data[i_face]
            elements = i_face_data['elements']  
            Dxi = i_face_data['Dxi']
            Dnu = i_face_data['Dnu']
            PNKSI = i_face_data['PNKSI']  
            
            Di = (mu*Dnu)/(PNKSI*Dxi)
            Fi = f [i_face]*rho*Dnu                             # Flux débitant donné
            
            bc_type_u , bc_value_u = self.bc_u [i_face]         # Type et valeur de condition au limite de la face pour la vitesse u
            bc_type_v , bc_value_v = self.bc_v [i_face]         # Type et valeur de condition au limite de la face pour la vitesse v
            
            if (bc_type_u == 'NEUMANN'):
                
                A_x [elements[0], elements[0]]  += Fi
                B_x [elements[0]] += mu * bc_value_u * Dnu - Fi * bc_value_u * PNKSI * Dxi
                      
            elif (bc_type_u == 'DIRICHLET'):
               
                A_x [elements[0], elements[0]]  += Di + max(Fi,0)
                B_x [elements[0]] += Di * bc_value_u + max(0,-Fi) * bc_value_u 
              
            if (bc_type_v == 'NEUMANN'):   
                
                A_y [elements[0], elements[0]]  += Fi
                B_y [elements[0]] += mu * bc_value_v * Dnu - Fi * bc_value_v * PNKSI*Dxi 
                
            elif (bc_type_v == 'DIRICHLET'):
              
                A_y [elements[0], elements[0]]  += Di + max(Fi,0)
                B_y [elements[0]] += Di*bc_value_v + max(0,-Fi)*bc_value_v 
          
        A = lil_matrix((n_elements,n_elements))  
        
        for i_face in range(self.mesh.get_number_of_boundary_faces(), n_faces):
            i_face_data = self.face_data[i_face]
            elements = i_face_data['elements']  
            Dnu = i_face_data['Dnu'] 
            PNKSI = i_face_data['PNKSI']
           
            Di = (mu*Dnu)/(PNKSI*i_face_data['Dxi'])
            Fi = f [i_face]*rho*Dnu                 # Flux débitant donné
            
            A [elements[0], elements[0]]  +=  Di
            A [elements[1], elements[1]]  +=  Di 
            A [elements[0], elements[1]]  -=  Di 
            A [elements[1], elements[0]]  -=  Di 
            
            if self.scheme == 'CENTRE' :
               
                A [elements[0], elements[0]]  +=  Fi/2
                A [elements[1], elements[1]]  -=  Fi/2
                A [elements[0], elements[1]]  +=  Fi/2
                A [elements[1], elements[0]]  -=  Fi/2
                
            if self.scheme == 'UPWIND' :
                A [elements[0], elements[0]]  +=  max(Fi,0)
                A [elements[1], elements[1]]  +=  max(0,-Fi)
                A [elements[0], elements[1]]  -=  max(0,-Fi)
                A [elements[1], elements[0]]  -=  max(Fi,0)
                
        A_x += A.copy() 
        A_x = A_x.tocsr()  
        
        A_y += A.copy()
        A_y = A_y.tocsr() 
        
        u = spsolve(A_x,B_x)      
        v = spsolve(A_y,B_y)
        
        return u, v, A_x
    
    def fonction_RhieChow (self, u, v, A_u, p, grad_p) :
        """
        Implémente la méthode de Rhie Chow pour calculer les flux débitants aux faces d'un maillage.
    
        Parameters
        ----------
        u : ndarray
            Champ de vitesse dans la direction x pour chaque élément.
        v : ndarray
            Champ de vitesse dans la direction y pour chaque élément.
        A_u : csr_matrix
            Matrice des coefficients utilisée pour résoudre les équations de la quantité de mouvement en u.
        p : ndarray
            Champ de pression pour chaque élément.
        grad_p : ndarray
            Gradient du champ de pression pour chaque élément.
    
        Returns
        -------
        uf : ndarray
            Tableau des flux débitants corrigés aux faces du maillage.
        """
        
        n_faces = self.mesh.get_number_of_faces()
        aire = self.elements_area                                       # Aire des éléments
        
        uf = np.zeros(n_faces)                                          # Initialisation des flux débitants
        A_u = A_u.toarray()
        
        DAU = np.diag(A_u)                                              # Termes diagonaux de la matrice ayant servit à construire u

        for i_face in range(self.mesh.get_number_of_boundary_faces()):
            i_face_data = self.face_data[i_face]
         
            bc_door_data = self.bc_door_data[i_face_data['tag']]        # Type de frontière entrée, paroi ou sortie
             
            if (bc_door_data == 'INLET' or bc_door_data == 'WALL'):     # On utilise les conditions aux limites sur les vitesses
                
                uf[i_face] = np.dot([self.bc_u[i_face][1], self.bc_v[i_face][1],], i_face_data['normal'])
             
            elif (bc_door_data == 'OUTLET'):                            # On utilise les conditions aux limites sur les pressions
                
                elements = i_face_data['elements']
                phi = [u [elements[0]], v [elements[0]]]
                
                uf[i_face] = np.dot(phi, i_face_data['normal']) 
                uf[i_face] += (aire[elements[0]]/DAU[elements[0]]) *((p[elements[0]]- self.bc_p[i_face][1] )/i_face_data['Dxi'] ) 
                uf[i_face] += np.dot(aire[elements[0]]*grad_p[elements[0]] /DAU[elements[0]] ,i_face_data['Exi']) 
                
        for i_face in range(self.mesh.get_number_of_boundary_faces(), n_faces):
            i_face_data = self.face_data[i_face]
            elements = i_face_data['elements']

            phi_x = (u[elements[0]]+ u[elements[1]])/2
            phi_y = (v[elements[0]]+ v[elements[1]])/2
            
            uf[i_face] += np.dot([phi_x, phi_y], i_face_data['normal']) 
            uf[i_face] += 0.5*((aire[elements[0]]/DAU[elements[0]]) + (aire[elements[1]]/DAU[elements[1]]))*((p[elements[0]]-p[elements[1]])/i_face_data['Dxi'])  
            uf[i_face] += 0.5*np.dot((aire[elements[0]]*grad_p[elements[0]] /DAU[elements[0]] + aire[elements[1]]*grad_p[elements[1]] /DAU[elements[1]]), i_face_data['Exi'])
            
        return uf
    def correction_pression(self,A_x,Uf):
        """
        Ajuste le champ de vitesse en fonction des corrections de pression.

        Paramètres
        ----------
        A_x : scipy.sparse matrix
            Matrice du momentum.
        Uf : ndarray
            Vecteur de vitesses initiales à chaque face du maillage.
        
        Retourne
        -------
        UfF : ndarray
            Vecteur de vitesses corrigées après application de la correction de pression.
        P : ndarray
            Vecteur de pressions calculées à chaque élément du maillage.
        np.sum(np.abs(B)) : float
            Divergence de la correction
        """
        n_elements = self.mesh.get_number_of_elements()  
        n_faces = self.mesh.get_number_of_faces()
        aire = self.elements_area
        A_x = A_x.toarray()
        UfF = np.zeros(n_faces)
        DAU = np.diag(A_x) 
        rho = self.coeff_data[0]
        
        NP = lil_matrix((n_elements,n_elements))
        B = np.zeros(n_elements)
        
        for i_face in range(self.mesh.get_number_of_boundary_faces()):
            i_face_data =  self.face_data[i_face]
            elements = i_face_data['elements']
            Dnu  = i_face_data['Dnu']
            
            bc_door_data = self.bc_door_data[i_face_data['tag']]
           
            if (bc_door_data == 'INLET' or bc_door_data == 'WALL'):
            
                B [elements[0]] -= rho*Uf[i_face]*Dnu
            
            if (bc_door_data == 'OUTLET'):
            
                Dfi = aire[elements[0]]/(DAU[elements[0]]*i_face_data['Dxi'])
                
                NP [elements[0], elements[0]]  +=  rho*Dfi*Dnu
                B [elements[0]] -= rho*Uf[i_face]*Dnu
                
        for i_face in range(self.mesh.get_number_of_boundary_faces(), n_faces):
            i_face_data =  self.face_data[i_face]
            elements = i_face_data['elements']   
            Dnu  = i_face_data['Dnu']
            

            Dfi = (1/(2*i_face_data['Dxi']))*((aire[elements[0]]/DAU[elements[0]]) + (aire[elements[1]]/DAU[elements[1]]))
            NP [elements[0], elements[0]]  +=  rho*Dfi*Dnu
            NP [elements[1], elements[1]]  +=  rho*Dfi*Dnu
            NP [elements[0], elements[1]]  -=  rho*Dfi*Dnu
            NP [elements[1], elements[0]]  -=  rho*Dfi*Dnu
            
            B [elements[0]] -= rho*Uf[i_face]*Dnu
            B [elements[1]] += rho*Uf[i_face]*Dnu
            
        NP = NP.tocsr()
        P = spsolve(NP,B) 
        for i_face in range(self.mesh.get_number_of_boundary_faces()):
            i_face_data =  self.face_data[i_face]
            
            bc_door_data = self.bc_door_data[i_face_data['tag']]
            
            if (bc_door_data == 'INLET' or bc_door_data == 'WALL'):
                
                UfF[i_face] = Uf[i_face]
                
            if (bc_door_data == 'OUTLET'):
                
                elements = i_face_data['elements']     

                Dfi = aire[elements[0]]/(DAU[elements[0]]*i_face_data['Dxi'])
                UfF[i_face] = Uf[i_face] + Dfi*P[elements[0]]
                
        for i_face in range(self.mesh.get_number_of_boundary_faces(), n_faces):
            i_face_data = self.face_data[i_face]
            elements = i_face_data['elements']  
            
            Dfi = (1/(2*i_face_data['Dxi']))*((aire[elements[0]]/DAU[elements[0]]) + (aire[elements[1]]/DAU[elements[1]]))
            UfF[i_face] = Uf[i_face] + Dfi*(P[elements[0]] - P[elements[1]]) 

        return UfF,P, np.sum(np.abs(B))
    
        
def init_face_data(mesh, bc_speed_data, bc_pressure_data):
    """
    Initialise les données géométriques et physiques associées à chaque face du maillage,
    et  les conditions aux limites pour les vitesses (u, v) et la pression (p) 
    sur les faces frontières du maillage.
    
    Parameters
    ----------
    mesh : object
        Objet représentant le maillage. 
    bc_speed_data : list
        Liste contenant les conditions aux limites pour les vitesses.
    bc_pressure_data : list
        Liste contenant les conditions aux limites pour la pression.
        
    Returns
    -------
    face_data : dict
        Dictionnaire contenant les données associées à chaque face du maillage.
    bc_u : ndarray structuré 
        Conditions aux limites pour la vitesse u.
    bc_v : ndarray structuré 
        Conditions aux limites pour la vitesse v.
    bc_p : ndarray structuré 
        Conditions aux limites pour la pression p.
    """
    n_faces = mesh.get_number_of_faces()
    n_boundary_faces = mesh.get_number_of_boundary_faces()
    elements_coords = mesh.elements_coords()
    
    dtype = [("type", "U10"), ("value", "f8")]          # Définir le type structuré
    
    face_data = {}
    bc_u = np.zeros(n_boundary_faces, dtype=dtype)      # Initialisation de u structuré des faces frontières
    bc_v = np.zeros(n_boundary_faces, dtype=dtype)      # Initialisation de v structuré des faces frontières
    bc_p = np.zeros(n_boundary_faces, dtype=dtype)      # Initialisation de p structuré des faces frontières
    
    for i_face in range(n_boundary_faces):
                                            
        elements = mesh.get_face_to_elements(i_face)    
        tag = mesh.get_boundary_face_to_tag(i_face)         
        nodes = mesh.get_face_to_nodes(i_face)
        
        Dnu = mesh.get_face_to_norm(i_face)
        
        xA, yA = mesh.get_nodes_to_xy_mean_coord(nodes)
        xT, yT = elements_coords[elements[0]]                       
        
        Dx = xA - xT
        Dy = yA - yT
      
        Dxi = np.sqrt((Dx)**2+(Dy)**2)
        
        normal = mesh.get_face_to_normal(i_face)

        Exi = np.array([Dx/Dxi, Dy/Dxi])
        PNKSI = np.dot(normal,Exi)
        
        face_data[i_face] = {
            'elements' : elements,
            'tag' : tag,
            'Dnu' : Dnu,
            'Dxi': Dxi,
            'Exi' : Exi,
            'normal' : normal,
            'PNKSI' : PNKSI,
            
        }
        # Remplir bc_u
        bc_type_u = bc_speed_data[tag][0][0]
        bc_value_u = 0
        if bc_type_u == "DIRICHLET":
            bc_value_u = bc_speed_data[tag][1][0](xA, yA)
        elif bc_type_u == "NEUMANN":
            bc_value_u = np.dot(bc_speed_data[tag][2][0](xA, yA), normal)
        bc_u[i_face] = (bc_type_u, bc_value_u)              # Type et valeur de condition au limite de la face pour la vitesse u
    
        # Remplir bc_v
        bc_type_v = bc_speed_data[tag][0][1]
        bc_value_v = 0
        if bc_type_v == "DIRICHLET":
            bc_value_v = bc_speed_data[tag][1][1](xA, yA)
        elif bc_type_v == "NEUMANN":
            bc_value_v = np.dot(bc_speed_data[tag][2][1](xA, yA), normal)
        bc_v[i_face] = (bc_type_v, bc_value_v)              # Type et valeur de condition au limite de la face pour la vitesse v
    
        # Remplir bc_p
        bc_type_p = bc_pressure_data[tag][0]
        bc_value_p = 0
        if bc_type_p == "DIRICHLET":
            bc_value_p = bc_pressure_data[tag][1](xA, yA)
        elif bc_type_p == "NEUMANN":
            bc_value_p = np.dot(bc_pressure_data[tag][2](xA, yA), normal)
        bc_p[i_face] = (bc_type_p, bc_value_p)              # Type et valeur de condition au limite de la face pour la pression p
        
    for i_face in range(mesh.get_number_of_boundary_faces(), n_faces):
        
        elements = mesh.get_face_to_elements(i_face)
        nodes = mesh.get_face_to_nodes(i_face)
        
        xA, yA = elements_coords[elements[1]]
        xP, yP = elements_coords[elements[0]]
        
        Dx = xA - xP 
        Dy = yA - yP
        
        Dxi = np.sqrt((Dx)**2+(Dy)**2)
        Dnu = mesh.get_face_to_norm(i_face)
        
        normal = mesh.get_face_to_normal(i_face)
        
        Exi = np.array([Dx/Dxi, Dy/Dxi])

        PNKSI = np.dot(normal,Exi)
        
        face_data[i_face] = {
            'elements' : elements,
            'tag' : tag,
            'Dnu' : Dnu,
            'Dxi': Dxi,
            'Exi': Exi,
            'normal' : normal,
            'PNKSI' : PNKSI,
        }
    return face_data, bc_u, bc_v, bc_p   
