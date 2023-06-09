import cv2


# Ambil input dari camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Membuat background substract
object_BS = cv2.createBackgroundSubtractorMOG2()


while True:
    # Baca frame menjadi bentuk numpy array
    ret, frameV = cap.read()

    mask = object_BS.apply(frameV)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv2.contourArea (cnt)
        if area > 120:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frameV, (x, y), (x + w, y+ h), (0, 0, 255), 2)
            detections.append([x, y, w, h])

    cv2.imshow("Deteksi Obyek", frameV)

    key = cv2.waitKey(15)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()        

