import cv2
import numpy as np


def filter_colors(image):
	# Cria uma matriz que representa a cor "branco fraco"
	lower_white = np.array([200, 200, 200], dtype=np.uint8)

	# Cria uma matriz que representa a cor "branco forte"
	upper_white = np.array([255, 255, 255], dtype=np.uint8)

	# Filtra as cores e deixa só o range das cores específicadas 
	white_mask = cv2.inRange(image, lower_white, upper_white)

	# Mescla imagem e aplica a mascara
	white_image = cv2.bitwise_and(image, image, mask=white_mask)

	# Filtra os pixes amarelos 
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

	lower_yellow = np.array([15, 38, 115], dtype=np.uint8)

	upper_yellow = np.array([35, 204, 255], dtype=np.uint8)

	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

	# Combina as duas imagens 
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

	return image2
