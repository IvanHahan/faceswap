import dlib
import cv2
import os
import numpy as np
from utils.path import make_dir_if_needed

# 1.creating a video object
video = cv2.VideoCapture(0)
# 2. Variable
a = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

make_dir_if_needed('data/my/images')
make_dir_if_needed('data/my/landmarks')

# 3. While loop
while True:
    a = a + 1
    # 4.Create a frame object
    check, frame = video.read()
    # Converting to grayscale
    shapes = detector(frame)
    for s in shapes:
        left = s.left() - 40
        right = s.right() + 40
        top = s.top() - 40
        bottom = s.bottom() + 40

        landmarks = predictor(frame, s)

        rec = len(os.listdir('data/my/images'))
        face_segment = frame[top:bottom, left:right].copy()

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 140), 2)
        landmarks_np = np.zeros((0, 2), dtype='int32')
        for i in range(68):
            p = landmarks.part(i)
            cv2.circle(frame, (p.x, p.y), 1, (0, 0, 255), -1)
            landmarks_np = np.append(landmarks_np, [[p.x - left, p.y - top]], axis=0)
        cv2.imwrite('data/my/images/{}.png'.format(rec), face_segment)
        np.save('data/my/landmarks/{}.npy'.format(rec), landmarks_np)


    # 5.show the frame!
    cv2.imshow("Capturing", frame)

    # 6.for playing
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# 7. image saving
showPic = cv2.imwrite("filename.jpg", frame)
print(showPic)
# 8. shutdown the camera
video.release()
cv2.destroyAllWindows()
