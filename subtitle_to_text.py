# base64 is an encoding method that converts invisible characters into visible characters
import base64
# opencv is a cross-platform computer vision library that implements many common algorithms in image processing and computer vision
import os
import cv2
import requests
# from aip import AipOcr
from PIL import Image
from pytesseract import pytesseract
import csv
import pandas as pd
import pandas as pd
import numpy as np
import re
import csv



def tailor_video():
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for word in r:
            input_txt = re.sub(word, "", input_txt)
        return input_txt

    path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.tesseract_cmd = path_to_tesseract
    video_path = os.path.join('song.mp4')
    times = 0
    frameFrequency = 10
    outPutDirName = './image/'
    if not os.path.exists(outPutDirName):
        # If the file directory does not exist, create a directory
        os.makedirs(outPutDirName)
    camera = cv2.VideoCapture(video_path)
    with open('output.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow("[Post]")
        while True:
            text2 = ""
            times += 1
            res, image = camera.read()
            if not res:
                break
            if times % frameFrequency == 0:
                cv2.imwrite(outPutDirName + str(times) + '.jpg', image)
                image_path = "./image/" + str(times) + '.jpg'
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img)
                text2 = text2 + text
            #print(text2)
            #print("    {}".format(text2))
            csvwriter.writerow([text2])  
    print('Picture extraction finished')
# organizing output
    df = pd.read_csv(r'output.csv', encoding= 'unicode_escape')
    df.dropna(subset = ["["], inplace=True)
    df['Post'] = np.vectorize(remove_pattern)(df['['].astype(str), "[&/@[\w]*]")
    df['Post'] = df['Post'].replace('[\n\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]','', regex=True)
    df.to_csv('list', sep='\t', index=False, line_terminator='\n')

    text = ' '.join(df['Post'].tolist())
    print(text)

# open the file in the write mode
    with open('Post.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([text])



