import face_recognition
import cv2
from PIL import Image, ImageDraw

# Env = minicurso

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

# Carrega a imagem de teste 
image = face_recognition.load_image_file("2021-05-15.png")

# Recebe as posições dos rostos
face_locations = face_recognition.face_locations(image)
# Faz a codificação para aprender como reconhece
face_encodings = face_recognition.face_encodings(image, face_locations)

# Cria uma imagem usando de base a array da original
pil_image = Image.fromarray(image)
# Crie uma instância ImageDraw para desenhar
draw = ImageDraw.Draw(pil_image)

for face_encoding in face_encodings:
        # Faz a verificação se existe algum dos rostos na imagem
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
        
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
            draw.rectangle(((left, top), (right, bottom)), outline = (0, 0, 255))

            text_width, text_height = draw.textsize(name)    

            # desenha o nome da pessoa na imagem
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill = (0, 0, 255), outline = (0, 0, 255))
            
            # Insere o texto na imagem
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


# Evitar uso de excessivo de memória 
del draw

# Salva a imagem alterada
pil_image.save("testeB.jpg")