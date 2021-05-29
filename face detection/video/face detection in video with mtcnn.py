import cv2
from mtcnn.mtcnn import MTCNN


def faceDectetion(name):
    # Contado de quantos rostos foram detectados
    count = 0

    # Recebe a função que faz a detecção
    detector = MTCNN()

    # 'cv2.VideoCapture' abre o vídeo
    cam = cv2.VideoCapture(name)

    # Loop infinito
    while True:
        # ret é uma variável booleana que retorna verdadeiro se o quadro estiver disponível.
        # frame é um vetor de matriz de imagens capturado com base nos quadros por segundo padrão definidos explícita ou implicitamente, referencia.
        # 'video.read()' retorna o próximo frame do vídeo.
        ret, frame = cam.read()

        if frame is not None:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # lista com os resultados
            result = detector.detect_faces(frame)

            if result != [] and round(result[0]['confidence'], 2) > 0.89:
                # Caso exista mais de um rostos detectado irá desenha o cotorno em todos
                for face in result: 
                    limit_face = face['box']

                    # Caminho e o nome da foto
                    name = "C:\\Users\\Felip\\source\\repos\\Mini curso\\face detection\\video\\mtcnn\\" + str(count) + '.jpg'
                   
                    # Desenha um retângulo em volta do rostos
                    cv2.rectangle(frame, (limit_face[0], limit_face[1]), (limit_face[0] + limit_face[2], limit_face[1] + limit_face[3]), (0, 155, 255), 2)

                    # Mostra grau de confidencia do resultado
                    print(result[0]['confidence'])

                    # Cria o arquivo (foto)
                    cv2.imwrite(name, frame)

                    count += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


# Caminho do vídeo original
faceDectetion("C:\\Users\\Felip\\source\\repos\\Mini curso\\videos\\faceDetection.mp4")
