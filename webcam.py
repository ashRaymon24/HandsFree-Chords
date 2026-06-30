import cv2

stream = cv2.VideoCapture(0)

if not stream.isOpened():
    print("No Stream")
    exit()

while True:
    ret, frame = stream.read()
    if not ret:
        print("No more stream")
        break
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord("q"): #stops webcam video when q key is pressed
        break

stream.release()
cv2.destroyAllWindows()