from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from OCRConfig import readNumber, ingDataInitial
import easyocr
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# 開発環境、本番環境でもapiエンドポイントは /api/ocr でOK
# フロントエンドかつ本番側ではapiエンドポイントをフルURLにする

@app.route("/api/ocr", methods=['POST'])
def ocr():
  files = request.files.getlist('image')
  ingData = ingDataInitial.copy()
  for file in files:
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = image.shape
    bottom_start = int(height * 0.05)
    cropped_image = image[bottom_start:, :]
    cv2.imwrite(f'croppedImg_{file.filename}.jpg', cropped_image)
    reader = easyocr.Reader(['en','ja'], gpu=False)
    detections = reader.readtext(cropped_image)
    os.remove(f'croppedImg_{file.filename}.jpg')

    numbers = []
    ings = []
    xFound = False
    for detection in detections:
      xAxisBottomRight = detection[0][2][0]
      yAxisBottomRight = detection[0][2][1]
      text = detection[1]
      if text.startswith("x"):
        try:
          number = int(text.replace("x", ""))
          numbers.append([number, xAxisBottomRight, yAxisBottomRight])
          xFound = True
        except ValueError:
          pass
      elif text in readNumber and xFound:
        ings.append([text, xAxisBottomRight, yAxisBottomRight])
    for i in range(len(ings)):
      mindiff = float('inf')
      minIndex = -1
      for j in range(len(numbers)):
        diff = (ings[i][1] - numbers[j][1])*(ings[i][1] - numbers[j][1]) + (ings[i][2] - numbers[j][2])*(ings[i][2] - numbers[j][2])
        if diff < mindiff:
          mindiff = diff
          minIndex = j
      if minIndex != -1:
        ingData[readNumber[ings[i][0]]] = numbers[minIndex][0]
  return jsonify(ingData)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)