import numpy as np
from mss import mss
import sys
import time
import cv2
from pynput.mouse import Button, Controller
import keyboard
import matplotlib.pyplot as plt

def new_construct_window(w, h):
    bound_box = {'top' : 0, 'left' : 0, 'width' : w, 'height' : h}
    screenshot = np.array(mss().grab(bound_box))
    return screenshot

def color_2_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_faces(gray_img):
    face_cascade = cv2.CascadeClassifier('C:/Users/Adam/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale( gray_img, scaleFactor = 1.1, minNeighbors = 3, minSize = (100,100) )
    return faces


def draw_faces(img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img


def detect_people(gray_img):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    people, _ = hog.detectMultiScale(gray_img, winStride=(10,10) )
    return people

def draw_people(img, people):
    for (x, y, w, h) in people:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    return img

def show_img(img):
    cv2.imshow('Window', img)
    time.sleep(0.017)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    while not (keyboard.is_pressed("=")):
        color_img = new_construct_window(1920,1080)
        gray_img = color_2_gray(color_img)
        faces = detect_faces(gray_img)
        people = detect_people(gray_img)
        finished_img = draw_faces(color_img, faces)
        finished_img = draw_people(color_img, people)
        show_img(finished_img)