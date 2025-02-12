"""
ORB-SIFT
Script que permite visualizar y obtener los puntos característicos a través de los algoritmos ORB y SIFT con 5 imágenes que han sido rotadas,
determinar corrimientos y generar Point Clouds.
Autor: José Gómez Mateus
Fecha: 12/02/2025
Contacto: https://github.com/jagomezmat 
          email: jagomezmat@unal.edu.co
Licencia: "Uso restringido para estudio académico"
"""

#Importaciones
#-------------------------------------------------------------------------------------------------------------------------

from cmath import cos, pi
import math
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
import csv

img0=cv2.imread('azucar003.tif',0) #imagen central o de referencia
img1=cv2.imread('azucar000.tif',0) #imagen ángulo +10
img2=cv2.imread('azucar006.tif',0) #imagen ángulo -10
img3=cv2.imread('azucar001.tif',0) #imagen ángulo +5
img4=cv2.imread('azucar005.tif',0) #imagen ángulo -5
#-------------------------------------------------------------------------------------------------------------------------


#Definición de constantes y variables globales:
#-------------------------------------------------------------------------------------------------------------------------
# Umbrales de desplazamientos verticales y horizontales para los keypoints entre imágenes
vertical_threshold = 10 # Establece tu umbral de desplazamiento vertical aquí
horizontal_thresholdmax = 35  # Establece tu umbral de desplazamiento horizontal aquí
horizontal_thresholdmin=0  # Establece tu umbral de desplazamiento horizontal mínimo aquí
topz=100000 #establecer umbral de altura en z

#ángulos de rotación
angulo1 = 20 #ángulo de rotación completa entre  tres imágenes
theta1 = math.radians(angulo1) #pasar a radianes
phi1 = (math.pi/2 - theta1/2) #obtener ángulo para cálculo de profundidad

angulo2 = 10 #ángulo de rotación completa entre  tres imágenes
theta2 = math.radians(angulo2) #pasar a radianes
phi2 = (math.pi/2 - theta2/2) #obtener ángulo para cálculo de profundidad
#-------------------------------------------------------------------------------------------------------------------------


#Definición de funciones
#-------------------------------------------------------------------------------------------------------------------------
#Función con los métodos de detección de matches entre imágenes y sus parámetros
def create_detector(name):
    if name == "ORB":
        return cv2.ORB_create(nfeatures=50000, scaleFactor=1.1, nlevels=19)
    elif name == "SIFT":
        return cv2.SIFT_create(nfeatures=50000, nOctaveLayers=10, contrastThreshold=0.03, edgeThreshold=20, sigma=1.1)
    elif name == "KAZE":
        return cv2.KAZE_create()
    elif name == "AKAZE":
        return cv2.AKAZE_create()
    else:
        raise ValueError(f"Detector desconocido: {name}")

#Función para ejecutar el proceso de detección de matches
def process_images(img1, img2, detector, vertical_threshold, horizontal_thresholdmax, horizontal_thresholdmin, return_keypoints=False):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    if detector.__class__ in [cv2.ORB, cv2.AKAZE]:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    matches = bf.match(des1, des2)
    filtered_matches = filter_matches_by_displacement(matches, kp1, kp2, vertical_threshold, horizontal_thresholdmax, horizontal_thresholdmin)#llamado a la función para filtrar
    
    if return_keypoints:
        return kp1, kp2, filtered_matches #permite obtener los kp si return_keypoints=True
    else:
        return filtered_matches

#Función para filtrar emparejamientos usando umbrales horizontales y verticales
def filter_matches_by_displacement(matches, kp1, kp2, vertical_threshold, horizontal_thresholdmax, horizontal_thresholdmin):
    """
    Filtra los emparejamientos basándose en umbrales de desplazamiento vertical y horizontal.

    :param matches: lista de emparejamientos obtenidos por BFMatcher.
    :param kp1: puntos clave de la primera imagen.
    :param kp2: puntos clave de la segunda imagen.
    :param vertical_threshold: umbral de desplazamiento vertical.
    :param horizontal_thresholdmax: umbral de desplazamiento horizontal.
    :return: lista filtrada de emparejamientos.
    """
    filtered_matches = []
    for match in matches:
        # Obtiene las posiciones x y y de los puntos clave emparejados
        y1, x1 = kp1[match.queryIdx].pt
        y2, x2 = kp2[match.trainIdx].pt
        
        # Comprueba si las diferencias verticales y horizontales son menores que los umbrales
        if abs(y1 - y2) < vertical_threshold and horizontal_thresholdmin<= abs(x1 - x2) < horizontal_thresholdmax:
            filtered_matches.append(match)
    
    return filtered_matches

#función para mostrar las imagenes con los matches
def show_matches(img1, img2, kp1, kp2, matches, title='Matches'):
    """
    Muestra las imágenes con los emparejamientos dibujados.

    :param img1: Primera imagen.
    :param img2: Segunda imagen.
    :param kp1: Puntos clave de la primera imagen.
    :param kp2: Puntos clave de la segunda imagen.
    :param matches: Emparejamientos a mostrar.
    :param title: Título de la ventana.
    """
    # Crear una máscara para mostrar solo los emparejamientos filtrados
    matches_mask = [1] * len(matches)
    
    # Dibujar los emparejamientos usando la máscara
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchesMask=matches_mask)
    
    # Mostrar la imagen con emparejamientos
    cv2.imshow(title, img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Función para guardar datos en csv
def save_to_csv(coords_dict, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Escribir encabezado
        csvwriter.writerow(['Combination', 'Image', 'X', 'Y'])
        
        # Recorrer el diccionario y escribir cada coordenada
        for combination, data in coords_dict.items():
            for img_name, coords in data.items():
                for x, y in coords:
                    csvwriter.writerow([str(combination), img_name, x, y])

#-------------------------------------------------------------------------------------------------------------------------

# Definir las imágenes y sus combinaciones
images = [img0, img1, img2, img3, img4] #img0 siempre será la imagen de referencia
combinations = [(0, 1), (0, 2), (1, 2),(0, 3), (0, 4), (3,4)] #solamente combinaciones de img0 a 1, 0 a 3 y de 1 a 2, el parámetro crossCheck=True hace que los emparejamientos sean reciprocos, por lo que no es necesario hacer de 1 a 2 y de 2 a 1

# Definir detectores
detectors = ["ORB","SIFT"]


# Procesar todas las combinaciones
for det_name in detectors:
    detector = create_detector(det_name)
    for idx1, idx2 in combinations:
        filtered_matches = process_images(images[idx1], images[idx2], detector, vertical_threshold, horizontal_thresholdmax,horizontal_thresholdmin)
        print(f"Number of filtered {det_name} matches between Image {idx1} and Image {idx2}: {len(filtered_matches)}")



#visualizar UNICAMENTE los keypoints de los emparejamientos filtrados
#crear diccionarios para guardar las coordenadas de los keypoints emparejados en las imagenes 
orb_coords = {}
sift_coords = {}
sum_methods_coords = {}


for det_name in detectors:
    detector = create_detector(det_name)
    for idx1, idx2 in combinations:
        kp1, kp2, filtered_matches = process_images(images[idx1], images[idx2], detector, vertical_threshold, horizontal_thresholdmax, horizontal_thresholdmin, return_keypoints=True)
        
        # Filtrar los keypoints basados en los emparejamientos filtrados
        matched_kp1 = [kp1[match.queryIdx] for match in filtered_matches]
        matched_kp2 = [kp2[match.trainIdx] for match in filtered_matches]

        # Crear nuevos objetos DMatch con índices actualizados
        new_matches = [cv2.DMatch(i, i, match.distance) for i, match in enumerate(filtered_matches)]
        
        #Guardar coordenadas
        if det_name == "ORB":
            orb_coords[(idx1, idx2)] = {
                'image1': [kp.pt for kp in matched_kp1],
                'image2': [kp.pt for kp in matched_kp2]
            }
        elif det_name == "SIFT":
            sift_coords[(idx1, idx2)] = {
                'image1': [kp.pt for kp in matched_kp1],
                'image2': [kp.pt for kp in matched_kp2]
            }

        #para mostrar las imagenes con los emparejamientos filtrados, comentar para agilizar el proceso                
        title = f"Filtered {det_name} matches between Image {idx1+1} and Image {idx2+1}"
        show_matches(images[idx1], images[idx2], matched_kp1, matched_kp2, new_matches, title)


# Combinar y filtrar coordenadas para cada combinación de imágenes
for (idx1, idx2) in combinations:
    # Combinar coordenadas de 'image1'
    combined_image1_coords = orb_coords[(idx1, idx2)]['image1'] + sift_coords[(idx1, idx2)]['image1']
    unique_image1_coords = list(set([tuple(coord) for coord in combined_image1_coords]))

    # Combinar coordenadas de 'image2'
    combined_image2_coords = orb_coords[(idx1, idx2)]['image2'] + sift_coords[(idx1, idx2)]['image2']
    unique_image2_coords = list(set([tuple(coord) for coord in combined_image2_coords]))

    # Guardar en el diccionario sum_methods_coords
    sum_methods_coords[(idx1, idx2)] = {
        'image1': unique_image1_coords,
        'image2': unique_image2_coords
    }


# Guardar coordenadas de los emparejamientos de cada imagen y mostrar su cantidad con ORB y SIFT
#si se desean obtener las coordenadas emparejadas de la image_ref con la imagen 1 usando ORB, se haría así:
#coords_img2_with_img1_orb = orb_coords[(0, 1)]['image1']
#De manera similar, se desean obtener las coordenadas emparejadas de la image_ref con la imagen 2 usando SIFT:
#coords_img2_with_img3_sift = sift_coords[(0, 2)]['image1']

combinations_names = {
    (0, 1): ("imagen ref", "imagen 1"),
    (0, 2): ("imagen ref", "imagen 2"),
    (1, 2): ("imagen 1", "imagen 2"),
    (0, 3): ("imagen ref", "imagen 3"),
    (0, 4): ("imagen ref", "imagen 4"),
    (3, 4): ("imagen 3", "imagen 4")
    
}

def print_matched_coordinates(detector_name, coords_dict):
    for (idx1, idx2), names in combinations_names.items():
        num_coords_img1 = len(coords_dict[(idx1, idx2)]['image1'])
        num_coords_img2 = len(coords_dict[(idx1, idx2)]['image2'])
        print(f"Número de coordenadas emparejadas en {names[0]} con {names[1]} usando {detector_name}: {num_coords_img1}")
        print(f"Número de coordenadas emparejadas en {names[1]} con {names[0]} usando {detector_name}: {num_coords_img2}")

print_matched_coordinates("ORB", orb_coords)
print_matched_coordinates("SIFT", sift_coords)
print_matched_coordinates("ORB+SIFT", sum_methods_coords)

# Guardar coordenadas en un archivo CSV
save_to_csv(orb_coords, 'coordinatesORB.csv')
save_to_csv(sift_coords, 'coordinatesSIFT.csv')
save_to_csv(sum_methods_coords, 'coordinatesORB+SIFT.csv')

# Extraer coordenadas de los keypoints emparejados usando metod_coords:sift_coords, orb_coords
#0 es la imagen de referencia, img1 es la imagen de angulo positivo, img2 es la pareja de angulo negativo
# Función para obtener coordenadas emparejadas, 0 siempre debe ser la imagen de referencia
def extract_keypoint_coordinates(metod_coords, img1, img2):
    list_kpimg1_ref = np.array(metod_coords[(0, img1)]['image2'])
    list_kpref_img1 = np.array(metod_coords[(0, img1)]['image1'])
    list_kpimg1_img2 = np.array(metod_coords[(img1, img2)]['image1'])
    list_kpimg2_img1 = np.array(metod_coords[(img1, img2)]['image2'])
    return list_kpimg1_ref, list_kpref_img1, list_kpimg1_img2, list_kpimg2_img1

# Función para calcular corrimientos
def calculate_shifts(coords1, coords2):
    return coords2 - coords1

def add_points_to_matrix(img1_img2_coords, img1_ref_coords, ref_img1_coords, shifts, phi, matrix=None):
    if matrix is None:
        matrix = []
    
    for i in range(len(img1_img2_coords)):
        for j in range(len(img1_ref_coords)):
            if (img1_img2_coords[i,0] == img1_ref_coords[j,0]) and (img1_img2_coords[i,1] == img1_ref_coords[j,1]) and (-10 < shifts[i,1] < 10):
                xp = ref_img1_coords[j,0]
                yp = ref_img1_coords[j,1]
                DeltaX = shifts[i,0]
                zp = (DeltaX/2) / math.cos(phi)
                if abs(zp.real) <= topz: # filtro sobre las altura z
                    matrix.append((xp, yp, zp.real))
    return matrix

#Iniciar matriz donde se van a guardar los points clouds
points = []

# Con método SIFT
# Coordenadas emparejadas entre imgref, img1 e img2
list_kp1_ref, list_kpref_1, list_kp1_2, list_kp2_1= extract_keypoint_coordinates(sift_coords, 1, 2)
# Calcular corrimientos entre img1 e img2
Corrimientos1a2 = calculate_shifts(list_kp1_2, list_kp2_1)
# Añadir puntos a la matriz principal para las imágenes 1 y 2
points = add_points_to_matrix(list_kp1_2, list_kp1_ref, list_kpref_1, Corrimientos1a2, phi1, matrix=points)

# Coordenadas emparejadas entre imgref, img3 e img4
list_kp3_ref, list_kpref_3, list_kp3_4, list_kp4_3= extract_keypoint_coordinates(orb_coords, 3, 4)
# Calcular corrimientos entre img3 e img4
Corrimientos3a4 = calculate_shifts(list_kp3_4, list_kp4_3)
# Añadir puntos a la matriz principal para las imágenes 3 y 4
points = add_points_to_matrix(list_kp3_4, list_kp3_ref, list_kpref_3,Corrimientos3a4, phi2, matrix=points)


# Con método ORB
# Coordenadas emparejadas entre imgref, img1 e img2
list_kp1_ref, list_kpref_1, list_kp1_2, list_kp2_1= extract_keypoint_coordinates(orb_coords, 1, 2)
# Calcular corrimientos entre img1 e img2
Corrimientos1a2 = calculate_shifts(list_kp1_2, list_kp2_1)
# Añadir puntos a la matriz principal para las imágenes 1 y 2
points = add_points_to_matrix(list_kp1_2, list_kp1_ref, list_kpref_1, Corrimientos1a2, phi1, matrix=points)

# Coordenadas emparejadas entre imgref, img3 e img4
list_kp3_ref, list_kpref_3, list_kp3_4, list_kp4_3= extract_keypoint_coordinates(orb_coords, 3, 4)
# Calcular corrimientos entre img3 e img4
Corrimientos3a4 = calculate_shifts(list_kp3_4, list_kp4_3)
# Añadir puntos a la matriz principal para las imágenes 3 y 4
points = add_points_to_matrix(list_kp3_4, list_kp3_ref, list_kpref_3,Corrimientos3a4, phi2, matrix=points)

# Después de añadir todos los puntos a 'points'
original_length = len(points)

# Convertir 'points' a un conjunto y luego de nuevo a una lista para eliminar duplicados
points = list(set(points))
length_after_removing_duplicates = len(points)

# Calcular el número de coordenadas repetidas
num_duplicates = original_length - length_after_removing_duplicates

print(f"El número de coordenadas repetidas es: {num_duplicates}")

#imprimir los resultados
print("pointsSIFT_ORB")
print(np.shape(points))

#grabar en ply
np.savetxt('PointsSIFT_ORB.csv', points, delimiter=';',fmt='%1.2f')
pcdP = o3d.geometry.PointCloud()
pcdP.points = o3d.utility.Vector3dVector(points)

o3d.io.write_point_cloud("./PointsSIFT_ORB.ply", pcdP, write_ascii=True)
o3d.visualization.draw_geometries([pcdP])
