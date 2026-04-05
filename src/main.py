import tkinter as tk
import cv2
from PIL import Image, ImageTk
import time


# ====== IMPORT FACE RECOGNITION CỦA BẠN ======
from src.detector import detect_faces
from src.recognizer import recognize_faces
from src.utils import draw_results


class FaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceID Management System - AI")
        self.root.geometry("900x700")

        # ====== STATE ======
        self.cap = None
        self.running = False

        # ====== UI ======
        self.setup_ui()

        self.unknown_images = []   # lưu ảnh (ImageTk)
        self.unknown_labels = []   # lưu label
        self.last_capture_time = 0
        self.frame_count = 0
        self.face_locations = []
        self.face_names = []

    # ================= UI =================
    def setup_ui(self):
        # ===== HEADER =====
        header = tk.Frame(self.root, bg="green", height=50)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)

        title = tk.Label(header, text="Face Recognizer",
                         bg="green", fg="white", font=("Time New Roman", 16))
        title.grid(row=0, column=0, padx=10, sticky="w")

        btn_start = tk.Button(header, text="Start", bg="green", fg="white",
                              width=8, command=self.start_cam)
        btn_stop = tk.Button(header, text="Stop", bg="red", fg="white",
                             width=8, command=self.stop_cam)
        # btn_setting = tk.Button(header, text="Setting", width=8)

        btn_start.grid(row=0, column=1, padx=5)
        btn_stop.grid(row=0, column=2, padx=5)
        # btn_setting.grid(row=0, column=3, padx=5)

        header.grid_columnconfigure(0, weight=1)

        # ===== BODY =====
        self.label_cam = tk.Label(self.root, bg="lightgray")
        self.label_cam.grid(row=1, column=0,columnspan=2, sticky="nsew", padx=5, pady=5)

        # side_panel = tk.Frame(self.root, bg="red")
        # side_panel.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        footer = tk.Frame(self.root, bg="white", height=200)
        footer.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=0, pady=0)

        # lưu ảnh
        self.list_nuknown = tk.Frame(footer, bg="white")
        self.list_nuknown.pack(fill="both",expand=True)

        # ===== GRID CONFIG =====
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=1)
        # self.root.grid_rowconfigure(2, weight=1)

    # ================= CAMERA =================
    def start_cam(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop_cam(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.label_cam.config(image="")

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            self.frame_count += 1
            if self.frame_count %5 == 0:
            # if ret:
                # ====== 1. FACE RECOGNITION ======
                face_locations, rgb_small = detect_faces(frame)
                face_names = recognize_faces(rgb_small, face_locations)
                frame = draw_results(frame, face_locations, face_names)
                
                    
                if "Me" in face_names:
                    # cv2.imwrite("me.jpg",frame)
                    if time.time() - self.last_capture_time > 2:  # 2 giây mới lưu 1 lần
                        self.add_img(frame)
                        self.last_capture_time = time.time()

                # ====== 2. FIT THEO UI ======
                w = self.label_cam.winfo_width()
                h = self.label_cam.winfo_height()

                if w > 0 and h > 0:
                    frame = cv2.resize(frame, (w, h))

                # ====== 3. CONVERT ĐỂ HIỂN THỊ ======
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)

                self.label_cam.imgtk = imgtk
                self.label_cam.configure(image=imgtk)

            # ====== LOOP ======
            self.label_cam.after(30, self.update_frame)

    def add_img(self,frame):
        face_unknown = cv2.resize(frame,(100,100))
        face_unknown = cv2.cvtColor(face_unknown,cv2.COLOR_BGR2RGB)

        t = time.strftime("%H:%M:%S")

        img = Image.fromarray(face_unknown) # đổi dl pixel thành dtg ảnh
        imgtk = ImageTk.PhotoImage(img) # chuyển định dạng sang tkinter

        self.unknown_images.append(imgtk)
        index = len(self.unknown_images)-1
        cols = 5
        row = index // cols
        col = index % cols

        label = tk.Label(self.list_nuknown,bg="white")
        label.grid(row=row, column=col, padx=5, pady=5)

        #ảnh + time
        img_label = tk.Label(label,image=imgtk)
        img_label.pack()

        time_label = tk.Label(label,text=t, font=("Arial",8))
        time_label.pack()

        self.unknown_labels.append(label)



# ===== RUN APP =====
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()

# import cv2
# from src.detector import detect_faces
# from src.recognizer import recognize_faces
# from src.utils import draw_results
# import tkinter 



# def main():
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         face_locations, rgb_small = detect_faces(frame) #lấy toàn bộ khuôn mặt trong frame
#         face_names = recognize_faces(rgb_small, face_locations) #nhận diện khuôn mặt trong db

#         frame = draw_results(frame, face_locations, face_names) # hiển thị, vẽ lên frame

#         cv2.imshow("Face Recognition", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()