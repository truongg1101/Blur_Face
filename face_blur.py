import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
import os

output_directory = "E:/Tgmt/output/image/"

def resize_image(img, width, height):
    return cv2.resize(img, (width, height))

def blur_faces():
    filepath = entry.get()
    image = cv2.imread(filepath)

    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    h, w = image.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 3
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    output = np.squeeze(model.forward())

    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        if confidence > 0.4:
            box = output[i, 3:7] * np.array([w, h, w, h])
            start_x, start_y, end_x, end_y = box.astype(int)
            face = image[start_y:end_y, start_x:end_x]
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            image[start_y:end_y, start_x:end_x] = face

    resized_original = resize_image(cv2.imread(filepath), 400, 600)
    resized_blurred = resize_image(image, 400, 600)

    cv2.imshow('Original Image', resized_original)
    cv2.imshow('Blurred Faces', resized_blurred)

    output_path = os.path.join(output_directory, os.path.basename(filepath))
    cv2.imwrite(output_path, image)

def browse_file():
    filepath = filedialog.askopenfilename()
    entry.delete(0, END)
    entry.insert(0, filepath)

root = Tk()
root.title("Blur Faces")

frame = Frame(root)
frame.pack(padx=50, pady=50)

label = Label(frame, text="Đường dẫn ảnh:")
label.grid(row=0, column=0)

entry = Entry(frame, width=40)
entry.grid(row=0, column=1, padx=10)

browse_button = Button(frame, text="Chọn ảnh", command=browse_file)
browse_button.grid(row=1, columnspan=2, pady=10)

blur_button = Button(frame, text="Làm mờ khuôn mặt", command=blur_faces)
blur_button.grid(row=2, columnspan=2, pady=10)

root.mainloop()
