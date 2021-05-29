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
    path = './videos/lane1_1.mp4'
    #path = './videos/final.mp4'
    
    temp = 0

    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        try:
            ret, frame = cap.read()

            hsv = filter_colors(frame)

            temp = 1

            blur = cv2.GaussianBlur(hsv, (5, 5), 0) # to reduce the noise 

            edges = cv2.Canny(blur, 50, 150) # to find the edges

            aoi = area_of_interest_video(edges)

            lines = cv2.HoughLinesP(aoi, 2, np.pi/180, 100, np.array([]), 20, 5)

            avg_lines= combo_lines(frame, lines)

            clines = show_lines(frame, avg_lines)

            color_image_line = cv2.addWeighted(frame, 0.9, clines, 1, 1)

            res = cv2.resize(color_image_line, (1280, 640))

            cv2.imshow('Window', res) # to show the outpout

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break # to quit press q

        except Exception:
            pass

    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    video()
