import numpy as np


def make_cordinates(image, parameter):
    try:
        slope, intercept = parameter
        # Recebe a altura
        y1 = image.shape[0]

        y2 = int(y1*(3.5/5))

        x1 = int((y1 - intercept)/slope)

        x2 = int((y2 - intercept)/slope)

    except Exception:
        slope, intercept = 0, 0

    # Cria e retorna a matriz
    return np.array([x1, y1, x2, y2])


def combo_lines(lane_image, lines):
    try:
        left_lane = []
        right_lane = []

        # Cria a matriz 
        temp_left = np.array([ 75, 720, 466, 503])
        temp_right = np.array([860, 720, 631, 503])

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            if y1 == y2:
                continue
            else:
                # Ajuste polinomial de mínimos quadrados
                para = np.polyfit((x1, x2), (y1, y2), 1)

                slope = para[0]

                intercept = para[1]

                if slope < 0:
                    left_lane.append((slope, intercept))
                else:
                    right_lane.append((slope, intercept))

        # Calculo da média ponderada no eixo especificado
        left_avg = np.average(left_lane, axis=0)
        right_avg = np.average(right_lane, axis=0)

        left_line = make_cordinates(lane_image, left_avg)
        right_line = make_cordinates(lane_image, right_avg)
        
    except Exception:
        pass

    # Cria e retorna a matriz
    return np.array([left_line, right_line])
