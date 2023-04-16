import base64
from io import BytesIO
import random

from flask import Flask, render_template, request

from PIL import Image
from PIL import ImageDraw

import predict

app = Flask("medic_predict")

@app.route('/', methods=['GET', 'POST'])
def upload():
    return render_template("upload_form.html")


@app.route('/upload_success', methods = ['POST'])
def success():
    koeff = float(request.form["koeff"])
    f = request.files['file']
    image = Image.open(f.stream)

    image_with_predict = predict.object_detection(image, predict.yolo3, koeff)

    buffer = BytesIO()
    image_with_predict.save(buffer, format="JPEG")
    buffer.seek(0)
    base64string = base64.b64encode(buffer.read()).decode('ascii')
    imgAsBase64 = "data:image/jpeg;base64,"+base64string
    return render_template("upload_success.html", name = f.filename, img_base_64=imgAsBase64)


@app.route('/upload_success_wo_predict', methods = ['POST'])
def success_wo_predict():
    f = request.files['file']
    # imagePath = "uploaded_files/"+f.filename
    # f.save(imagePath)
    # print('type(f)', type(f))
    image = Image.open(f.stream)
    image = image.convert('RGB')
    drawing = ImageDraw.Draw(image)

    N = 15
    x1 = random.sample(range(0, image.width), N)
    x2 = random.sample(range(0, image.width), N)
    y1 = random.sample(range(0, image.height), N)
    y2 = random.sample(range(0, image.height), N)

    rectangles = zip(x1, y1, x2, y2)
    for xy in rectangles:
        x1 = xy[0]
        y1 = xy[1]
        x2 = xy[2]
        y2 = xy[3]

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        drawing.rectangle((x1, y1, x2, y2), width=3,
                          outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    # drawing.rectangle((1, 3, 10, 15), outline=(255, 55, 5), width=3)

    # updatedFileName = "updated_files/"+f.filename+'.jpeg'
    # image.save(updatedFileName, 'JPEG')
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    base64string = base64.b64encode(buffer.read()).decode('ascii')
    imgAsBase64 = "data:image/jpeg;base64,"+base64string
    # print(base64string, '\n', '-'*15)
    # print(imgAsBase64, '\n', '-'*15)
    return render_template("upload_success.html", name = f.filename, img_base_64=imgAsBase64)
    #
    # return render_template("upload_success.html", name = f.filename, img_base_64="")

