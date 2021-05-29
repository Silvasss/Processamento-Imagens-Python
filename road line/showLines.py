import cv2
import numpy as np


def show_lines(img, lines):
    # Retorne uma matriz de zeros com a mesma forma e tipo 
    line_image = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            # Inicializar as funções na ordem certa
            try:
                x1, y1, x2, y2 = line.reshape(4)

                if(x1 > 2000 or x2 > 2000):
                    x1 = 1920
                    x2 = 1920
                    # Altera a matriz adicionando a cor verde
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
                else:
                    # Altera a matriz adicionando a cor verde
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

            except Exception:
                pass

    return line_image
