
import urllib
import cv2

import easyocr as ocr
import numpy as np

reader = ocr.Reader(['en'])
def process_image(url=None,path=None):
	if url != None:
		image = url_to_image(url)
	elif path != None:
		image = cv2.imread(path)
	else:
		return "Invalid path"

	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	_,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	dst = cv2.fastNlMeansDenoising(th2, 10, 10, 7)
	print ("Recognising...")
	rec_string =  reader.readtext(dst, detail=0)
	print ("the result is {}".format(rec_string))
	return rec_string

def url_to_image(url):
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image
