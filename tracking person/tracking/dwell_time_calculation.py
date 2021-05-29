import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker


protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
# Carrega o modelo  para a memória
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

#Roadline2

# Inicializa a lista de objetos que o modelo foi treinado para detectar 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Framework simples e eficiente de rastreamento e detecção de objetos
tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

# Marcação 
def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        # Identifica o tipo geral do dado.
        if boxes.dtype.kind == "i":
            # Cópia da matriz, convertida 
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        # Ordem
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            # Tamanho da matriz
            last = len(idxs) - 1

            # Recebe a ultima posição da matriz
            i = idxs[last]

            pick.append(i)

            # Retorna o valor mais alto entre as listas
            #print(i ,x1[i], x1[idxs[:last]])
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])

            # Retorna o menor valor entre as listas
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
                        
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Sobreposição
            overlap = (w * h) / area[idxs[:last]]
            
            # Apaga o ultimo elemento da matriz
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
            
        return boxes[pick].astype("int")

    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


def main():
    # 'cv2.VideoCapture' abre o vídeo
    cam = cv2.VideoCapture('train-final.mp4')

    object_id_list = []
    dtime = dict()
    dwell_time = dict()

    # Loop infinito
    while True:
        # ret é uma variável booleana que retorna verdadeiro se o quadro estiver disponível.
        # frame é um vetor de matriz de imagens capturado com base nos quadros por segundo padrão definidos explícita ou implicitamente, referencia.
        # 'video.read()' retorna o próximo frame do vídeo.
        ret, frame = cam.read()

        # Redimensiona para a largura/altura mantendo a proporção
        frame = imutils.resize(frame, width=600)

        # Retorna o número de linhas e colunas
        (H, W) = frame.shape[:2]

        # Cria um blob quadridimensional a partir da imagem
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        # Chama o Modelo 
        detector.setInput(blob)
        person_detections = detector.forward()

        rects = []
        
        for i in np.arange(0, person_detections.shape[2]):            
            confidence = person_detections[0, 0, i, 2]

            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])

                (startX, startY, endX, endY) = person_box.astype("int")

                rects.append(person_box)

        # Cria uma array 
        boundingboxes = np.array(rects)

        boundingboxes = boundingboxes.astype(int)

        rects = non_max_suppression_fast(boundingboxes, 0.2)

        objects = tracker.update(rects)

        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            if objectId not in object_id_list:
                object_id_list.append(objectId)
                dtime[objectId] = datetime.datetime.now()
                dwell_time[objectId] = 0
            else:
                curr_time = datetime.datetime.now()

                old_time = dtime[objectId]

                time_diff = curr_time - old_time

                dtime[objectId] = datetime.datetime.now()

                sec = time_diff.total_seconds()

                dwell_time[objectId] += sec


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            text = "{}|{}".format(objectId, int(dwell_time[objectId]))

            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)


        cv2.imshow("Track ID", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


main()
