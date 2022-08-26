from msilib.schema import File
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import PIL.ExifTags


import tensorflow.keras as keras
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.layers import Input, Conv2D, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow import lite

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = keras.models.load_model("mod.h5")
Interpreter = lite.Interpreter(model_path="litemod.tflite")
Interpreter.allocate_tensors()


def evaluate_tflite_model(dataset, interpreter):

    #     print(interpreter.get_input_details())

    input_index_0 = interpreter.get_input_details()[0]["index"]
    input_index_1 = interpreter.get_input_details()[1]["index"]
    input_index_2 = interpreter.get_input_details()[2]["index"]
    input_index_3 = interpreter.get_input_details()[3]["index"]

    output_index = interpreter.get_output_details()[0]["index"]

    predictions = []
    for (image, eyecrop, fincrop, scalecrop), label in dataset.unbatch().take(dataset.unbatch().cardinality()):

        image = np.expand_dims(image, axis=0).astype(np.float32)
        eyecrop = np.expand_dims(eyecrop, axis=0).astype(np.float32)
        fincrop = np.expand_dims(fincrop, axis=0).astype(np.float32)
        scalecrop = np.expand_dims(scalecrop, axis=0).astype(np.float32)

        label = np.argmax(label)

        interpreter.set_tensor(input_index_0, image)
        interpreter.set_tensor(input_index_1, eyecrop)
        interpreter.set_tensor(input_index_2, fincrop)
        interpreter.set_tensor(input_index_3, scalecrop)
        interpreter.invoke()

        output = interpreter.tensor(output_index)
        output = output()[0]
        predictions.append(output)

    return predictions


@app.post("/img")
async def root(file: UploadFile = File(...)):
    with open("saved_images/destination.png", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    X = ["saved_images/destination.png"]

    eyexml = cv2.CascadeClassifier('./cascades/eye.xml')
    finxml = cv2.CascadeClassifier('./cascades/fin.xml')
    scalexml = cv2.CascadeClassifier('./cascades/cascade.xml')

    def create_image_embs(img_path):
        img_path = str(img_path.numpy().decode())
        img = Image.open(img_path)

        img = img.resize((256, 256))
        print(img.size)
        img = np.array(img)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        # cv2.imshow("img",img)
        # try:
        #     gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        #     print("all")
        # except:
        #     gray = np.array(img)
        #     img = np.expand_dims(img,axis=-1)
        #     print('else')

        eyes = eyexml.detectMultiScale(gray)
        fins = finxml.detectMultiScale(gray)
        scales = scalexml.detectMultiScale(gray)

        if len(eyes):
            x, y, w, h = eyes[0]
            eyecrop = img[x:x+w, y:y+h]
            if (w != 32):
                eyecrop = Image.fromarray(
                    eyecrop.astype(np.uint8)).resize((32, 32))
        else:
            eyecrop = np.zeros((32, 32, 3))

        if len(fins):
            x, y, w, h = fins[0]
            fincrop = img[x:x+w, y:y+h]
            if (w != 32):
                fincrop = Image.fromarray(
                    fincrop.astype(np.uint8)).resize((32, 32))
        else:
            fincrop = np.zeros((32, 32, 3))

        if len(scales):
            x, y, w, h = scales[0]
            scalecrop = img[x:x+w, y:y+h]
            if (w != 32):
                #             print(w)
                scalecrop = Image.fromarray(
                    scalecrop.astype(np.uint8)).resize((32, 32))
        else:
            scalecrop = np.zeros((32, 32, 3))

        img = tf.convert_to_tensor(np.float32(np.array(img)))
        eyecrop = tf.convert_to_tensor(np.float32(np.array(eyecrop)))
        fincrop = tf.convert_to_tensor(np.float32(np.array(fincrop)))
        scalecrop = tf.convert_to_tensor(np.float32(np.array(scalecrop)))

        emb = (img, eyecrop, fincrop, scalecrop)

        return emb

    def tf_create_image_embs(img_path):
        [img, eyecrop, fincrop, scalecrop, ] = tf.py_function(
            create_image_embs, [img_path], [tf.float32, tf.float32, tf.float32, tf.float32, ])
        emb = (img, eyecrop, fincrop, scalecrop)
        return emb

    imageset = tf.data.Dataset.from_tensor_slices(X)
    imageset = imageset.map(tf_create_image_embs)
    print(type(imageset))
    print(imageset)
    y_ = [0]
    caty = to_categorical(tf.convert_to_tensor(y_), num_classes=9)
    caty = tf.data.Dataset.from_tensor_slices(caty)

    totalset = tf.data.Dataset.zip((imageset, caty))

    # MODEL = keras.models.load_model("mod.h5")
    ans = evaluate_tflite_model(totalset.batch(1), Interpreter)
    print(ans[0])
    ans1 = np.argmax(ans[0])

    classlist = ["Black Sea Sprat",
                 "Gilt-Head Bream",
                 "Hourse Mackerel",
                 "Red Mullet",
                 "Red Sea Bream",
                 "Sea Bass",
                 "Shrimp",
                 "Striped Red Mullet",
                 "Trout"]

    desc = [
        {"max_length": "14.5 cm", "common_length": "10.0 cm", "max_reported_age": "5 years",
            "max_published_weight": "5 g", "commonly_found": "Indian Ocean"},
        {"max_length": "70 cm", "common_length": "35.0 cm", "max_reported_age": "11 years",
            "max_published_weight": "17.2 kg", "commonly_found": "Bay of Bengal"},
        {"max_length": "80 cm", "common_length": "45.0 cm", "max_reported_age": "5 years", "max_published_weight": "4 kg",
            "commonly_found": "South-East coast and West coast of india,Kerala and Tamil Nadu coasts"},
        {"max_length": "40 cm", "common_length": "25.0 cm", "max_reported_age": "11 years",
            "max_published_weight": "1 kg", "commonly_found": "Goa,Konkan and Maharashtra coasts"},
        {"max_length": "100 cm", "common_length": "30.0 cm", "max_reported_age": "26 years",
            "max_published_weight": "9.7 kg", "commonly_found": "Anna Nagar, Chennai, Tamil Nadu"},
        {"max_length": "210 cm", "common_length": "80.0 cm", "max_reported_age": "30 years",
            "max_published_weight": "100 kg", "commonly_found": "Kerela,Tamil Nadu and many coastal areas of India"},
        {"max_length": "17 cm", "common_length": "8 cm", "max_reported_age": "6 years", "max_published_weight": "60 g",
            "commonly_found": "Kerala & Karnataka Coasts,Coasts of Andhra Pradesh & Odisha"},
        {"max_length": "40 cm", "common_length": "25 cm", "max_reported_age": "11 years",
            "max_published_weight": "1 Kg", "commonly_found": "Mediterranean Sea"},
        {"max_length": "140 cm", "common_length": "72 cm", "max_reported_age": "38 years", "max_published_weight": "50 Kg",
            "commonly_found": "Jammu and Kashmir (JK), Himachal Pradesh (HP), Uttarakhand (UK), Sikkim, Arunachal Pradesh (Ar. P), Western Ghats of Tamil Nadu (TN) and Kerala and, to some extent, in North Bengal, Manipur, Meghalaya and Nagaland"},
    ]

    try:
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in Image.open(X[0])._getexif().items()
            if k in PIL.ExifTags.TAGS
        }["GPSInfo"]
        ncords = "N:" + str(exif[2])
        ecords = "E:" + str(exif[4])
    except:
        ncords = "Unavailable"
        ecords = "Unavailable"

    if (ans[0][ans1] < 0.5):
        classn = "Unrecognized or No Fish"
        classid = "None"
        classd = "None"

    else:
        classn = str(classlist[ans1])
        classd = desc[ans1]
        classid = str(ans1)

    # img = cv2.imread("saved_images/destination.png", cv2.IMREAD_UNCHANGED)

    return {"filename": file.filename,
            "outputs": {"class_id": classid, "class_name": classn, "class_details": classd,
                        "GPS_cords_N": ncords, "GPS_cords_E": ecords,
                        "confidence": str(ans[0][ans1]), "conf_list": str(ans[0])}
            }
    # return {"file_name": file.filename, "size": file.filesize }
