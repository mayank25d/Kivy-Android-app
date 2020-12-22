"""
Facial Recognition - Mobile Version, June - July 2020.

"""
# from android import activity, mActivity
# from android.permissions import request_permissions, Permission
# from android.storage import primary_external_storage_path
# request_permissions([Permission.CAMERA, Permission.READ_EXTERNAL_STORAGE])

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.core.window import Window

from keras.models import load_model
import numpy as np
import PIL as pil
import cv2

# Person_000000 is "Unknown", Person_000001 is "Adiba" etc.
names = ["Unknown", "Adiba", "Aziza", "Rashid"]
model = load_model('model_VGGFace.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class FaceRecognitionApp(App):

    def build(self):
        self.width = 640
        self.height = 480
        layout = BoxLayout(orientation="vertical")
        self.img1 = Image()
        layout.add_widget(self.img1)
        self.but1 = Button(text="Start app", size_hint=(None, None), size=(200, 100))
        self.but1.bind(on_press=self.start_kivycam)
        layout.add_widget(self.but1)
        self.camera = self.root.ids['kivy_cam']
        return layout

    def start_kivycam(self, event):
        Clock.schedule_interval(self.update, 1.0/33.0)

    def update(self, dt):
        self.frame_texture = self.camera.texture
        # print(self.frame_texture.pixels)
        frame = np.frombuffer(self.frame_texture.pixels, dtype=np.uint8)
        # print(frame, frame.dtype, frame.shape)
        frame = np.reshape(frame, (self.height, self.width, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        # print(frame, frame.dtype, frame.shape)
        # frame = cv2.flip(frame, 0)

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        # (h, w) = frame.shape[:2]
        # center = (w / 2, h / 2)
        # M = cv2.getRotationMatrix2D(center, 90, 1.0)
        # frame = cv2.warpAffine(frame, M, (h, w))

        if faces is (): #syntax warning "is" with a literal. Did you mean "=="?
            faces = None
        if type(faces) is not np.ndarray:
            cv2.putText(frame, "No Face Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cropped_face = frame[y: y + h, x: x + w]
                cropped_face = cv2.resize(cropped_face, (224, 224))

                im = pil.Image.fromarray(cropped_face, 'RGB')
                img_array = np.array(im)
                img_array = np.expand_dims(img_array, axis=0)

                pred = model.predict(img_array)
                print(pred)

                name = names[0]
                for i in range(1, len(names)):
                    if pred[0][i] > 0.984 and pred[0][0] < 0.001:
                        name = names[i]
                    elif pred[0][0] > 0.5:
                        name = names[0]

                cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        # print(buf)
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture1.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img1.texture = texture1

if __name__ == '__main__':
    FaceRecognitionApp().run()
    cv2.destroyAllWindows()
