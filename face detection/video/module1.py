import cv2, os
 
img_array = []
 
for count in range(819):
    filename = 'C:\\Users\\Felip\\source\\repos\\Mini curso\\face detection\\video\\opencv\\' + str(count) + '.jpg'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
out = cv2.VideoWriter('opencv.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()