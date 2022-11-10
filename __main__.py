import cv2
import matplotlib.pyplot as plt
from gtts import gTTS
from pygame import mixer
from tempfile import TemporaryFile

# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions


class WebCamObjectDetection:
    def __init__(self, url=0):
        self.model = VGG19()
        self.capture = cv2.VideoCapture(url)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.capture.set(cv2.CAP_PROP_FRAME_COUNT, 1)
        self.capture.set(cv2.CAP_PROP_MODE, 1)

    def capture_frame(self):
        ret, self.frame = self.capture.read()
        self.frame = self.frame[8:-8, 48:-48, :]

    def detect_object(self):
        x = preprocess_input(self.frame.reshape((1, 224, 224, 3)))
        predictions = self.model.predict(x)
        _, self.top_predictions, _ = decode_predictions(predictions, 1)[0][0]
        print(self.top_predictions)

    def show_image(self):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(self.frame)
        ax.set_title(f"prediction: {self.top_predictions}")
        plt.show()

    def play_audio(self):
        tts = gTTS(text=self.top_predictions)
        file = TemporaryFile()
        tts.write_to_fp(file)
        file.seek(0)
        mixer.init()
        mixer.music.load(file)
        mixer.music.play()

    def run(self, n=5):
        for i in range(n):
            self.capture_frame()
            self.detect_object()
            self.show_image()


if __name__ == "__main__":
    # rtsp = 'rtsp://user:pass@192.168.1.85:6667/blinkhd'
    # http = 'http://192.168.1.85/axis-cgi/mjpg/video.cgi'
    WebCamObjectDetection().run()
