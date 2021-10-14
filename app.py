from flask import Flask , render_template , request , jsonify
from moviepy.video.VideoClip import VideoClip
import category
import matplotlib.pyplot as plt
from image_text import image_to_text
from PIL import Image
import numpy as np
from negative_post_analysis import nagative_post_analysis
from page_clustering import page_clustering
from subtitle_to_text import tailor_video
from video_to_text import video_to_text
from werkzeug.utils import secure_filename
import moviepy.editor as mp
import os



app = Flask(__name__)


@app.route("/")
def subtitleToText():
    # page_clustering()
    return "Hi"

@app.route("/sss")
def hello(): 
    return 'hey'
global pre
@app.route("/predict", methods =["POST"])
def categories():
    if request.method == "POST":
            name = request.get_json()
            cate_pred = category.category_prediction(name)
            for x in cate_pred:
                x
            pre=x
    return jsonify({'prediction': pre})
# negative post analysis

@app.route("/negativePostAnanlysis", methods =["POST"])
def negativePostAnalysis():
    if request.method == "POST":
        name = request.get_json()
        cate_pred = nagative_post_analysis(name)
        # for x in cate_pred:
        #     x
        # pre2=x
        print(cate_pred)
    return cate_pred

@app.route("/imageToText/", methods = ["POST"])
def imageToTex():
    if request.method == "POST":
        file = request.files['image']
        pil_image = Image.open(file)
        image = np.array(pil_image)
        # image = image / 255
        # image = image.reshape(-1, 32, 32, 3)
        text = image_to_text(image)
        print(text)
    return text

@app.route("/videoToText/", methods = ["POST"])
def videoToText():
     if request.method == "POST":
         file = request.files['video']
         # clean the filename
         safe_filename = secure_filename(request.files["video"].filename)
        #  filename = secure_filename(file.filename)
         request.files["video"].save(os.path.join("./videos/", safe_filename))
         video_clip = mp.VideoFileClip(os.path.join("./videos/", safe_filename))
        #  pil_image = VideoClip.open(file)
        #  image = np.array(pil_image)
        # image = image / 255
        # image = image.reshape(-1, 32, 32, 3)
         text = video_to_text(video_clip)
         print(text)
     return text
    



#url for mobile 
#  host='192.168.8.100',port=5000
if __name__ == "__main__":
    app.run(debug=False, host='192.168.8.101',port=5000)

