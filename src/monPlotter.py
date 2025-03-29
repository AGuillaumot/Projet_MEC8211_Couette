# -*- coding: utf-8 -*-
"""
MEC8211 - Vérification et Validation en modélisation numérique
Date de création: 2024 - 03 - 29
Auteur: Alban GUILLAUMOT
"""
import matplotlib.pyplot as plt
import numpy as np

from pyvistaqt import BackgroundPlotter
import pyvista as pv
import pyvistaqt as pvQt
from src.meshPlotter import MeshPlotter

class MonPlotter:
    def __init__(self, mesh, geometry):
        self.mesh = mesh
        self.geometry = geometry
        
        return
    def afficher_vitesses_rhie_chow(self, velocities, title):
        """
        Affiche les vitesses débitantes calculées par l'algorithme de Rhie-Chow.
    
        Parameters
        ----------
        velocities : ndarray
            Tableau des vitesses (scalaires) sur chaque arête du maillage.
        title : str, optional
            Titre du graphique (default is "Vitesses de Rhie-Chow avant correction").
    
        Returns
        -------
        None
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
    
        # Tracer les arêtes du maillage
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x_coords = [self.mesh.get_node_to_xcoord(nodes[0]), self.mesh.get_node_to_xcoord(nodes[1])]
            y_coords = [self.mesh.get_node_to_ycoord(nodes[0]), self.mesh.get_node_to_ycoord(nodes[1])]
            ax.plot(x_coords, y_coords, color='blue', linewidth=0.5)
    
        # Ajouter les vecteurs de vitesse sur chaque arête
        for i_face in range(self.mesh.get_number_of_faces()):
            xA, yA = self.mesh.get_nodes_to_xy_mean_coord(self.mesh.get_face_to_nodes(i_face))  # Coordonnées du centre de la face
            nx, ny = self.mesh.get_face_to_normal(i_face)      # Vecteur normal à la face
            velocity = velocities[i_face]                 # Vitesse calculée sur la face
    
            # Calculer les composantes du vecteur vitesse 
            vx = velocity * nx * 0.2  # Réduire la taille des vecteurs pour éviter qu'ils soient trop grands
            vy = velocity * ny * 0.2
    
            # Tracer le vecteur vitesse avec une échelle ajustée
            ax.quiver(xA, yA, vx, vy, angles='xy', scale_units='xy', scale=1, color='k')
    
        # Ajouter le titre et les limites des axes
        ax.set_title(title)
        dx = (self.geometry[1]-self.geometry[0])*0.1
        dy = (self.geometry[3]-self.geometry[2])*0.1
        ax.set_xlim(self.geometry[0]-dx, self.geometry[1]+dx)
        ax.set_ylim(self.geometry[2]-dy, self.geometry[3]+dy)
    
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(False)
        plt.show()
    def my_plotter_3D(self, u, v, titre):
        """
        Plot sur pyvista un champ vectoriel avec un fond basé sur la composante u et des flèches orientées par v et w.
        Colore les éléments où les composantes corrigées sont sous la tolérance en cyan.
        """
        plotter = MeshPlotter()
        # Préparation des données pour PyVista
        nodes, elements = plotter.prepare_data_for_pyvista(self.mesh)
        pv_mesh = pv.PolyData(nodes, elements)
        
        
        # Ajout des composantes
        phi_uv = np.column_stack((u, v, np.zeros(len(v))))
        pv_mesh["Vecteurs uv [m/s]"] = phi_uv
        
        norms = np.linalg.norm(phi_uv, axis=1)
        norms[norms == 0] = 1
        phi_uv_normalized = phi_uv / norms[:, np.newaxis]
        pv_mesh[titre] = phi_uv_normalized
        pv_mesh['scalars'] = np.sqrt(u**2 + v**2)
       
        pv_mesh_arrows = pv_mesh.copy()
        
        pl = BackgroundPlotter()
        
        # Ajout du maillage principal
        pl.add_mesh(pv_mesh, show_edges=True, color="blue", line_width=1, style="wireframe")
    
        arrows = pv_mesh_arrows.glyph(orient=titre, scale="scalars", factor=0.2)
        pl.add_mesh(arrows, color="black")
        # Supprimer l'affichage de la grille, mais garder les graduations et légendes
        
        pl.show_bounds(grid=False, location="outer", ticks="both", xtitle="X axis", ytitle="Y axis", minor_ticks=True, font_size=12)
        pl.add_text(titre, font_size=10, position='upper_edge')
        
        return
    def my_plotter_1D(self, phi, boolean):
        """
        Fonction qui permet la représentation d'un champ scalaire avec ses lignes de coupe et qui renvoie les coordonnées et valeurs des éléments sur les plans de coupe
    
        Parameters
        ----------
        phi : ndarray
            Champ scalaire que l'on souhaite représenter.
        titre : string
            Titre du graphique représentant le champ scalaire
    
        Returns
        -------
        ndarray
            Positions et valeurs du champ sur les 2 coupes.
    
        """
               
        elements_coords = self.mesh.elements_coords()
        boundary_nodes = self.mesh.get_boundary_faces_nodes(2)
        points = []
        
        n_elements = self.mesh.get_number_of_elements()
        for i_element in range(n_elements):
         
            element_nodes = self.mesh.get_element_to_nodes(i_element)
    
            if np.any(np.isin(element_nodes, boundary_nodes)):
                x, y = elements_coords[i_element]
                points.append([y, phi[i_element]])
                
        if boolean == True : 
            points.append([self.geometry[2],0])
            points.append([self.geometry[3],0])
        points = sorted(points, key=lambda x: x[0])
        
        points = np.array(points)
        return points
    
    def my_plotter_2D(self, phi, titre):
        """
        Fonction qui permet la représentation d'un champ scalaire avec ses lignes de coupe et qui renvoie les coordonnées et valeurs des éléments sur les plans de coupe
    
        Parameters
        ----------
        phi : ndarray
            Champ scalaire que l'on souhaite représenter.
        titre : string
            Titre du graphique représentant le champ scalaire
    
        Returns
        -------
        ndarray
            Positions et valeurs du champ sur les 2 coupes.
    
        """
        plotter = MeshPlotter()
        # Préparation des données pour PyVista
        nodes, elements = plotter.prepare_data_for_pyvista(self.mesh)
    
        # Création du maillage PyVista
        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh[titre] = phi

         # Déterminer les valeurs min et max de l'échelle
        min_phi, max_phi = np.min(phi), np.max(phi)
        
        # Créer une plage d'échelle pour clim avec des ticks tous les 0.1
        clim_range = (min_phi, max_phi)
        
        # Création du plotter PyVistaQt
        pl = BackgroundPlotter()
        
        # Ajouter le maillage avec une carte de couleurs
        pl.add_mesh(pv_mesh, show_edges=True, scalars=titre, cmap="viridis", clim=clim_range, scalar_bar_args={
            'title': "",               # Pas de titre pour la barre d'échelle
            'vertical': True,          # Barre d'échelle verticale
            'fmt': "%.1f",             # Format des références numériques (1 décimale)
            'n_labels': int((max_phi - min_phi) / 0.1) + 1,  # Nombre de ticks basés sur 0.1 d'intervalle
            'position_x': 0.875,        # Position horizontale (proche du bord droit)
            'position_y': 0.1,         # Position verticale (centrée sur le graphique)
            'width': 0.1,             # Largeur relative de la barre d'échelle
            'height': 0.8,             # Hauteur relative de la barre d'échelle
        })
    
        # Supprimer l'affichage de la grille, mais garder les graduations et légendes
        pl.show_bounds(grid=None, location="outer", ticks="both", xtitle="X axis", ytitle="Y axis", font_size=12)
        
        pl.add_text(titre, font_size=10, position='upper_edge')
        pl.show()
        return
