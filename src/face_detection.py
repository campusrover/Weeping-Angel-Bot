import numpy as np
from mss import mss
import sys
import time
import cv2
from pynput.mouse import Button, Controller
import keyboard
import matplotlib.pyplot as plt

def to_rgb(cref):
    mask = 0xff
    R = (cref & mask) / 255
    G = ((cref >> 8) & mask) / 255
    B = ((cref >> 16) & mask) / 255
    return(R, G, B)

def new_construct_window(bound_box):
    screenshot = np.array(mss().grab(bound_box))
    # array = plt.imread(screenshot)
    return screenshot


def detect_faces(w,h, face_cascade):
    x = (1920-w) // 2
    y = (1080-h) // 2
    bound_box = {'top' : y, 'left' : x, 'width' : w, 'height' : h}
    window = new_construct_window(bound_box)
    gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale( gray, scaleFactor = 1.1, minNeighbors = 3, minSize = (20,20) )
    for (x, y, w, h) in faces:
        cv2.rectangle(window,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('window', window)
    time.sleep(0.017)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('C:/Users/Adam/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    while not (keyboard.is_pressed("=")):
        start = time.perf_counter()
        coords = detect_faces(1920, 1080, face_cascade)
        print(coords)
