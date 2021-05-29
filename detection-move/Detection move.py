import cv2

# A váriavel que vai receber o frame estático
static_background = None

# 'cv2.VideoCapture' abre o vídeo
video = cv2.VideoCapture("teste.avi")

# Loop infinito
while True:
	# ret é uma variável booleana que retorna verdadeiro se o quadro estiver disponível.
    # frame é um vetor de matriz de imagens capturado com base nos quadros por segundo padrão definidos explícita ou implicitamente, referencia.
    # 'video.read()' retorna o próximo frame do vídeo.
	ret, frame = video.read()

	# Finaliza o código quando o vídeo terminar
	if not ret:
		break

	# Faz a conversão de BGR para GRAY.
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Faz a suavização para reduzir os ruidos
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# Altara o fundo para o primeiro frame do vídeo
	if static_background is None:
		static_background = gray
		continue

	# Faz o calulo da diferença entre a imagem estatica e o atual frame
	diff_frame = cv2.absdiff(static_background, gray)

	# Aplicar threshold caso o quadro atual for maior que 30, ele mostrará a cor branca
	thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]

	# Aumenta o thereshold para cobrir os pontos pretos
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)

	# Encontra os contornos do objeto
	cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		# Filtra os contornos
		if cv2.contourArea(contour) < 10000:
			continue

		# Posições
		(x, y, w, h) = cv2.boundingRect(contour)

		# Desenha um retângulo em volta do objeto
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


	# Mostra a imagem em preto e branco
	cv2.imshow("Threshold Frame", thresh_frame)

	# Mostra a imagem com os cotornos
	cv2.imshow("Color Frame", frame)

	if cv2.waitKey(1) == ord('q'):
		break


video.release()
cv2.destroyAllWindows()
