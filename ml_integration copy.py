from msilib.schema import File
import pandas as pd
from fastapi import FastAPI, UploadFile, File
import shutil
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()

credits_df= pd.read_csv("tmdb_5000_credits.csv")

students={
    1:{'name':'john',
       'age':17
    }
}


def location(col, n):
    l=[]
    for i in credits_df[col]:
        l.append(i)
    return(l[n])
    


#api
@app.get("/")
async def root():
    return {"greeting":"Hello world"}

@app.get("/{column}/{no}")
def getdata(column:str, no:int):
    if column=='title':
        return {'movie_name':(location(column, no))}
    elif column=='movie_id':
        return {'movie_id':(location(column, no))}
    else:
        return {'data':(location(column, no))}

@app.post("/img")
async def root(file: UploadFile = File(...)):
    with open("saved_images/destination.png", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    X = ["saved_images/destination.png"]

    eyexml = cv2.CascadeClassifier('./cascades/eye.xml')
    finxml = cv2.CascadeClassifier('./cascades/fin.xml')

    def create_image_embs(img_path):
        img_path = str(img_path.numpy().decode())
        img= Image.open(img_path)
        img=img.resize((256,256))
        img=np.array(img)
        try:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
            print("all")
        except:    
            gray = np.array(img)
            img = np.expand_dims(img,axis=-1)
            print('else')

        eyes = eyexml.detectMultiScale(gray)
        fins = eyexml.detectMultiScale(gray)
        
        if len(eyes):
            x,y,w,h = eyes[0]
            eyecrop = img[x:x+w, y:y+h]
        else:
            eyecrop=np.zeros((32,32,3))
        
        if len(fins):
            x,y,w,h = fins[0]
            fincrop = img[x:x+w, y:y+h]
        else:
            fincrop=np.zeros((32,32,3))
        
    #     img = tf.convert_to_tensor(np.array(img))
    #     eyecrop = tf.convert_to_tensor(np.array(eyecrop))
    #     fincrop = tf.convert_to_tensor(np.array(fincrop))
        
    #     print(eyecrop.shape)
        emb = (img,eyecrop,fincrop)
        
        return emb

    def tf_create_image_embs(img_path):
        [img, eyecrop, fincrop,] = tf.py_function(create_image_embs, [img_path], [tf.float32, tf.float32, tf.float32])
        emb = (img,eyecrop,fincrop) 
        return emb

    imageset = tf.data.Dataset.from_tensor_slices(X)
    imageset = imageset.map(tf_create_image_embs)

    # img = cv2.imread("saved_images/destination.png", cv2.IMREAD_UNCHANGED)
    dimensions=[]
    for img,eyecrop,fincrop in imageset.take(1):
        img_shape=img.shape
        dimensions.append(str(img.shape))
        print(eyecrop.shape)
        eye_crop=eyecrop.shape
        print(eye_crop)
        dimensions.append(str(eyecrop.shape))
        print(fincrop.shape)
        dimensions.append(str(fincrop.shape))
        fin_crop=fincrop.shape
        print(dimensions)
    
    return {"filename": file.filename, 
            "dimension":{"img_shape": str(img_shape), "eye_crop":str(eye_crop), "fin_crop":str(fin_crop)},

            }
    #return {"file_name": file.filename, "size": file.filesize }
