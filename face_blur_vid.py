import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
import os

output_directory = "E:/Tgmt/output/video/"

def blur_faces():
    filepath = entry.get()

    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    vid_capture = cv2.VideoCapture(filepath)

    if not vid_capture.isOpened():
        print("Could not open video file.")
        return

    fps = int(vid_capture.get(cv2.CAP_PROP_FPS))
    width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resized_width = int(width * 0.6)
    resized_height = int(height * 0.6)
    window_name = 'Blurred Faces in Video'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, resized_width, resized_height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = os.path.join(output_directory, os.path.basename(filepath))
    output_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    while True:
        ret, frame = vid_capture.read()

        if not ret:
            break

        h, w = frame.shape[:2]
        kernel_width = (w // 7) | 1
        kernel_height = (h // 7) | 1

        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        model.setInput(blob)
        output = np.squeeze(model.forward())

        for i in range(0, output.shape[0]):
            confidence = output[i, 2]
            if confidence > 0.4:
                box = output[i, 3:7] * np.array([w, h, w, h])
                start_x, start_y, end_x, end_y = box.astype(int)
                face = frame[start_y:end_y, start_x:end_x]
                face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
                frame[start_y:end_y, start_x:end_x] = face

        output_video.write(frame)

        resized_frame = cv2.resize(frame, (resized_width, resized_height))
        cv2.imshow(window_name, resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid_capture.release()
    output_video.release()
    cv2.destroyAllWindows()

def browse_file():
    filepath = filedialog.askopenfilename()
    entry.delete(0, END)
    entry.insert(0, filepath)

root = Tk()
root.title("Blur Faces in Video")

frame = Frame(root)
frame.pack(padx=50, pady=50)

label = Label(frame, text="Đường dẫn video:")
label.grid(row=0, column=0)

entry = Entry(frame, width=40)
entry.grid(row=0, column=1, padx=10)

browse_button = Button(frame, text="Chọn video", command=browse_file)
browse_button.grid(row=1, columnspan=2, pady=10)

blur_button = Button(frame, text="Làm mờ khuôn mặt trong video")
blur_button.grid(row=2, columnspan=2, pady=10)
blur_button.config(command=blur_faces)

root.mainloop()
