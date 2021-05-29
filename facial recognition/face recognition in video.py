import face_recognition
import cv2

# Env = minicurso

# 'cv2.VideoCapture' abre o vídeo
input_movie = cv2.VideoCapture("aula magda.mp4")

# Numero de frames do vídeo
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Cria um vídeo
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Usar a mesma resolução e frame rate do original
output_movie = cv2.VideoWriter('output.avi', fourcc, 59.97, (1920, 1080))

# Carrega a foto do rosto
cristina_image = face_recognition.load_image_file("./fotos/cristina.png")
# Faz a codificação para aprender como reconhece o rosto
cristina_face_encoding = face_recognition.face_encodings(cristina_image)[0]

fabiano_image = face_recognition.load_image_file("./fotos/fabiano.png")
fabiano_face_encoding = face_recognition.face_encodings(fabiano_image)[0]

fabio_image = face_recognition.load_image_file("./fotos/fabio.png")
fabio_face_encoding = face_recognition.face_encodings(fabio_image)[0]

fernanda_image = face_recognition.load_image_file("./fotos/fernanda.png")
fernanda_face_encoding = face_recognition.face_encodings(fernanda_image)[0]

jackson_image = face_recognition.load_image_file("./fotos/jackson.png")
jackson_face_encoding = face_recognition.face_encodings(jackson_image)[0]

madia_image = face_recognition.load_image_file("./fotos/madia.png")
madia_face_encoding = face_recognition.face_encodings(madia_image)[0]

parcilene1_image = face_recognition.load_image_file("./fotos/parcilene1.png")
parcilene1_face_encoding = face_recognition.face_encodings(parcilene1_image)[0]

parcilene2_image = face_recognition.load_image_file("./fotos/parcilene2.png")
parcilene2_face_encoding = face_recognition.face_encodings(parcilene2_image)[0]

# Lista com as codificações
known_faces = [
    cristina_face_encoding,
    fabiano_face_encoding,
    fabio_face_encoding,
    fernanda_face_encoding,
    jackson_face_encoding,
    madia_face_encoding,
    parcilene1_face_encoding,
    parcilene2_face_encoding
]

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

# Loop infinito
while True:
    # ret é uma variável booleana que retorna verdadeiro se o quadro estiver disponível.
    # frame é um vetor de matriz de imagens capturado com base nos quadros por segundo padrão definidos explícita ou implicitamente, referencia.
    # 'video.read()' retorna o próximo frame do vídeo.
    ret, frame = input_movie.read()

    # Quantos frames foram lidos
    frame_number += 1
    
    # Verifica se o vídeo não acabou
    if not ret:
        break
    
    # Recebe as posições dos rostos
    face_locations = face_recognition.face_locations(frame)

    # Faz a codificação para aprender como reconhece
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        # Faz a verificação se existe algum dos rostos na imagem
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
        
        name = None
        if match[0]:
            name = "Cristina"
        elif match[1]:
            name = "Fabiano"
        elif match[2]:
            name = "Fabio"
        elif match[3]:
            name = "Fernanda"
        elif match[4]:
            name = "Jackson"
        elif match[5]:
            name = "Madia"
        elif match[6]:
            name = "Parcilene1"
        elif match[7]:
            name = "Parcilene2"

        face_names.append(name)
    
        # Desenha os resultados
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Desenha um quadrado no rosto 
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Desenha o nome da pessoa na imagem
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    
    print("Writing frame {} / {}".format(frame_number, length))

    # Adicionando o frame ao novo vídeo criado
    output_movie.write(frame)


input_movie.release()
cv2.destroyAllWindows()