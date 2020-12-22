from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera

import cv2
import numpy as np

class Camrotate(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("chal ja")

class FaceRecognitionApp(App):
    def build(self):
        return Camrotate()

# class Cam(Camera):
#     def build(self):
#         pass
#
#     def get_texture(self):
#         return self.texture
#
#
# class CameraApp(App):
#     def build(self):
#         self.snapshot = None
#         root = BoxLayout()
#         self.width = 640
#         self.height  = 480
#         self.camera = Cam(resolution=(self.width, self.height), play=True)
#         root.add_widget(self.camera)
#         button = BoxLayout(orientation='vertical')
#         sn = Button(text='snapshot', pos=(self.width/2, 20), halign='center', size_hint=(None, None), size=(200,100))
#         sn.bind(on_press=self.get_texture1)
#         button.add_widget(sn)
#         root.add_widget(button)
#         return root
#
#     def get_texture1(self, event):
#         self.snapshot = self.camera.get_texture()
#         frame = np.frombuffer(self.snapshot.pixels, dtype=np.uint8)
#         frame = np.reshape(frame, (self.height, self.width, 4))
#         cv2.imshow("CV2", cv2.flip(frame, 0))
        
if __name__=='__main__':
    FaceRecognitionApp().run()
