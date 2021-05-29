import pytesseract, cv2

#Env = roadline2

# Carrega a imagem
img = cv2.imread("print.png")

while True:
    # Mostra a imagem
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

# Chama o programa instalado 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"

print(pytesseract.image_to_string(img))