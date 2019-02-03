import cv2
import time
import numpy as np
from PIL import Image
from gtts import gTTS
from pygame import mixer
from tempfile import TemporaryFile
from resizeimage import resizeimage
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


class WebCamObjectDetection:

	def __init__(self, url):
		self.url = url
		self.image_captured = None
		self.image_cropped = None
		self.predictions = None

	def capture(self):
		cap = cv2.VideoCapture(self.url)
		ret, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		self.image_captured = Image.fromarray(frame)

	def resize(self):
		self.image_cropped = resizeimage.resize_cover(self.image_captured, [224, 224])

	def show_image(self, image_size=0):
		img = [self.image_cropped, self.image_captured]
		img[image_size].show()

	def detect_object(self):
		frame = np.asarray(self.image_cropped.getdata())
		frame.resize((1, 224, 224, 3))
		img = preprocess_input(frame)

		# Generate predictions
		model = VGG16()
		predication = model.predict(img)
		self.predictions = decode_predictions(predication)[0]

	def play_audio(self, lang='en'):
		text = self.predictions[0][1].replace('_', ' ')
		print(text)
		tts = gTTS(text=text, lang=lang)

		file = TemporaryFile()
		tts.write_to_fp(file)
		file.seek(0)

		mixer.init()
		mixer.music.load(file)
		mixer.music.play()


if __name__ == "__main__":
	rtsp = 'rtsp://user:pass@192.168.1.85:6667/blinkhd'
	http = 'http://192.168.1.85/axis-cgi/mjpg/video.cgi'
	for i in range(1):
		od = WebCamObjectDetection(rtsp)
		od.capture()
		od.resize()
		od.show_image()
		od.detect_object()
		od.play_audio()
	time.sleep(5)
