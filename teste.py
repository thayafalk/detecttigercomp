import cv2 as cv

camera = cv.VideoCapture("tigres2.mp4")
#2IH2F7SXSTFS.jpg
cascade = cv.CascadeClassifier("treinamento/cascade.xml")
while True:
    #imagem = cv.imread("chicken/test/TDFAKEMSPE9G.jpg")
    _, imagem = camera.read()
    gray = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    objetos = cascade.detectMultiScale(gray, 1.25, 5)

    for (x,y,w,h) in objetos:
        cv.rectangle(imagem, (x,y),(x+w,y+h),(0,0,255),2)

    cv.imshow("tigre", imagem)
    k = cv.waitKey(60)
    if k == 27:
        break
    
cv.destroyAllWindows()
camera.release()