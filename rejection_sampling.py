import numpy as np 
import cv2


h = 300
w = 500
window = np.zeros((h, w), np.uint8)
prob = np.zeros(w, np.int16)

i = 0
while(1):
    i += 1
    rand1 = np.random.randint(w)
    rand2 = np.random.randint(w)
    if rand1 < rand2:
        prob[rand1] += 1
        window[h - prob[rand1] : h, rand1] = 255

    if i % 50 == 0:
        i = 0
        cv2.imshow("rejection_sampling", window)
        k = cv2.waitKey(1)
        if k == 27:
            break


cv2.destroyAllWindows()