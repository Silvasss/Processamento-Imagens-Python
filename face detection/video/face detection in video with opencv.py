import cv2


def faceDectetion(name):
    # Contado de quantos rostos foram detectados
    count = 0

    # 'cv2.VideoCapture' abre o vídeo
    video = cv2.VideoCapture(name)

    # Modelo pré-treinado
    cascPath = "haarcascade_frontalface_default.xml"   
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Loop infinito
    while True:
        # ret é uma variável booleana que retorna verdadeiro se o quadro estiver disponível.
        # frame é um vetor de matriz de imagens capturado com base nos quadros por segundo padrão definidos explícita ou implicitamente, referencia.
        # 'video.read()' retorna o próximo frame do vídeo.
        ret, frame = video.read()

        if frame is not None:
            # Faz a conversão de BGR para GRAY.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Retorna as posições dos rostos
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor = 1.1,  # Em uma foto de grupo, pode haver alguns rostos próximos à câmera do que outros. Naturalmente, esses rostos pareceriam mais proeminentes do que os de trás. Este fator compensa isso.
                minNeighbors = 3,   # Este parâmetro especifica o número de vizinhos que um retângulo deve ter para ser chamado de face.
                #minSize = (30, 30)  # Tamano minimo para detecção
            )

        # Verifica se o vídeo não acabou
        if ret:
            for (x, y, w, h) in faces:
                # Desenha um retângulo em volta do rostos
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Caminho e o nome da foto
                name = "C:\\Users\\Felip\\source\\repos\\Mini curso\\face detection\\video\\opencv\\" + str(count) + '.jpg'

                # A cada detecção e mostrado na tela o valor do count
                print(str(count))

                count += 1

                # Cria o arquivo (foto)
                cv2.imwrite(name, frame)
        else:
            break

    cam.release()
    cv2.destroyAllWindows()


# Caminho do vídeo original
faceDectetion("C:\\Users\\Felip\\source\\repos\\Mini curso\\videos\\faceDetection.mp4")
