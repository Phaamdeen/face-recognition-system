import tkinter as tk
import datetime
import os
import cv2
from PIL import Image, ImageTk
import util
import face_recognition
import numpy as np

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry('1200x5320+350+100')

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'blue', self.login)
        self.login_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

        # Track registered usernames and their face encodings
        self.registered_usernames = {}
        self.load_registered_usernames()

    def load_registered_usernames(self):
        for filename in os.listdir(self.db_dir):
            if filename.endswith('.jpg'):
                username = filename[:-4]  # Remove the .jpg extension
                image_path = os.path.join(self.db_dir, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                self.registered_usernames[username] = encoding

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture_arr = frame
            img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)
            self._label.after(20, self.process_webcam)
        else:
            print("Error: Could not read from webcam.")

    def login(self):
        unknown_img_path = './.tmp.jpg'
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)

        unknown_image = face_recognition.load_image_file(unknown_img_path)
        unknown_encoding = face_recognition.face_encodings(unknown_image)

        if not unknown_encoding:
            util.msg_box("Ups...", 'No face detected. Please try again.')
            os.remove(unknown_img_path)
            return

        unknown_encoding = unknown_encoding[0]
        face_distances = []
        usernames = []

        for username, encoding in self.registered_usernames.items():
            distance = face_recognition.face_distance([encoding], unknown_encoding)[0]
            face_distances.append(distance)
            usernames.append(username)

        # Find the closest match with a strict threshold
        if face_distances:
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            if best_distance < 0.4:  # Lower threshold to make it more strict
                recognized_username = usernames[best_match_index]
                util.msg_box('Welcome back!', f'Welcome, {recognized_username}.')
                with open(self.log_path, 'a') as f:
                    f.write(f'{recognized_username},{datetime.datetime.now()}\n')
            else:
                util.msg_box("Ups...", 'Unknown user. Please register or try again.')
        else:
            util.msg_box("Error", "No faces found in the database.")

        os.remove(unknown_img_path)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry('1200x5320+350+100')

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'blue',
                                                                      self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=600)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try Again',
                                                                         'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=700)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window,
                                                                'Please \ninput Fullname: ')
        self.text_label_register_new_user.place(x=750, y=70)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=400)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window,
                                                                'Please \ninput matric number: ')
        self.text_label_register_new_user.place(x=750, y=320)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()

        # Check if username already exists
        if name in self.registered_usernames:
            util.msg_box('Ups...', 'Username already exists: {}.'.format(name))
            return

        # Encode the new user's face
        new_user_image = face_recognition.face_encodings(self.register_new_user_capture)

        if not new_user_image:
            util.msg_box('Error', 'No face found. Please try again.')
            return

        new_user_encoding = new_user_image[0]

        # Check if the new user's face is too similar to an existing face
        for username, encoding in self.registered_usernames.items():
            distance = face_recognition.face_distance([encoding], new_user_encoding)[0]
            if distance < 0.45:  # Lower threshold for stricter similarity check
                util.msg_box('Ups...', f'Face already registered as: {username}.')
                return

        # Save the captured image and encoding
        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(name)), self.register_new_user_capture)
        self.registered_usernames[name] = new_user_encoding  # Add to the dict

        util.msg_box('Success!', 'User was registered successfully!')
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()


