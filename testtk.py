import tkinter as tk
import cv2
from PIL import Image, ImageTk

cap = None
running = False

def start_cam():
    global cap,running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        update_frame()

def update_frame():
    global cap,running

    if running :
        ret,frame = cap.read()
        if ret:
            # Resize cho đẹp
            frame = cv2.resize(frame, (600, 400))

            # Convert màu
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert sang Image Tkinter
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Hiển thị
            frame_cam.imgtk = imgtk
            frame_cam.configure(image=imgtk)

        # Lặp lại sau 10ms
        frame_cam.after(10, update_frame)

def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
    frame_cam.config(image="")

root = tk.Tk()
root.title("FaceID Management System - AI")
root.geometry("890x750")

# Tạo header frame màu xanh lá
header = tk.Frame(root, bg="green", height=50)
header.grid(row=0, column=0, columnspan=2, sticky="ew")  # chiếm full ngang
header.grid_propagate(False)  # giữ chiều cao cố định

# Cho cột 0 của root co giãn
root.grid_columnconfigure(0, weight=1)

# Thêm title và button vào header
title = tk.Label(header, text="Face Recognizer", bg="green", fg="white", font=("Time New Roman", 16))
title.grid(row=0, column=0, padx=10,pady=10, sticky="w")  # căn trái

Start = tk.Button(header, text="Start", bg="green", fg="white",width=8,command=start_cam)
Stop = tk.Button(header, text="Stop", bg="red", fg="white",width=8,command=stop_camera)
Setting = tk.Button(header, text="Setting", bg="green", fg="white",width=8)

# Đặt button bên phải header
Start.grid(row=0, column=1, padx=5, pady=5)
Stop.grid(row=0, column=2, padx=5, pady=5) 
Setting.grid(row=0, column=3, padx=5, pady=5)

# Cho header co giãn theo cột để button không bị dồn
header.grid_columnconfigure(0, weight=1)

# Frame nội dung phía dưới
frame_cam = tk.Label(root, text="VIDEO", bg="lightgray")
frame_cam.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

# Cho frame nội dung co giãn


footer = tk.Label(root,background="red")
footer.grid(row=1,column=1,sticky="nsew", padx=5, pady=5)
root.grid_columnconfigure(1, weight=1)

footer2 = tk.Label(root,background="blue",height=10)
footer2.grid(row=2,columnspan=2,sticky="nsew", padx=5, pady=5)
root.grid_columnconfigure(1, weight=1)
root.mainloop()
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     cap.release()
#     cv2.destroyAllWindows()