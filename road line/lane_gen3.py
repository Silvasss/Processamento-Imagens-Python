import cv2
import numpy as np
import time
from showLines import show_lines
from show_combo_lines import combo_lines
from showFilters import filter_colors
import matplotlib.pyplot as plt


def area_of_interest_video(img):
    # Recebe a altura e largura da imagem
    ht = img.shape[0]
    wt = img.shape[1]
    
    # Cria uma array com área de interesse
    triangle = np.array([[(0, ht-60), (wt, ht-60), (740, 420), (540, 420)]])

    # Retorne uma matriz de zeros com a mesma forma e tipo 
    mask = np.zeros_like(img)

    # Preenche a área com polígonos.
    cv2.fillPoly(mask, triangle, 255) 

    masked_image = cv2.bitwise_and(img, mask) # it will hide other data and show only the visible part

    return masked_image


def video():    
    # 'cv2.VideoCapture' abre o vídeo
    cam = cv2.VideoCapture('./videos/final.mp4')

    # Loop infinito
    while True:        
        # ret é uma variável booleana que retorna verdadeiro se o quadro estiver disponível.
        # frame é um vetor de matriz de imagens capturado com base nos quadros por segundo padrão definidos explícita ou implicitamente, referencia.
        # 'video.read()' retorna o próximo frame do vídeo.
        ret, frame = cam.read()
        
        # Finaliza o código quando o vídeo terminar
        if not ret:
            break

        # Inicializar as funções na ordem certa
        try:
            hsv = filter_colors(frame)

            # Faz a suavização para reduzir os ruidos
            blur = cv2.GaussianBlur(hsv, (5, 5), 0)

            # Função para encontrar as bordas na imagem
            edges = cv2.Canny(blur, 50, 150)

            # Função retorna a área de interesse no vídeo
            aoi = area_of_interest_video(edges)

            # Detecta as linhas na imagem
            lines = cv2.HoughLinesP(aoi, 2, np.pi/180, 100, np.array([]), 20, 5)

            # Chama a função que retorna a média das linhas de interesse
            avg_lines= combo_lines(frame, lines)

            # Chama a função que faz o desenho das linhas na imagem
            clines = show_lines(frame, avg_lines)

            # Junta as duas imagem e só uma
            color_image_line = cv2.addWeighted(frame, 0.9, clines, 1, 1)

            # Redimensionar a imagem final
            res = cv2.resize(color_image_line, (1280, 640))

            # Mostra o vídeo
            cv2.imshow('Video', res)

            if cv2.waitKey(1) == ord('q'):
                break     

        except Exception:
            pass
        

    cam.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    video()