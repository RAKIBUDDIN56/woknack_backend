from PIL import Image
from pytesseract import pytesseract
from langdetect import detect_langs
from googletrans import Translator, constants
from pprint import pprint


def image_to_text(img_eng):
   # image_path_eng = (r"C:\Users\user\Desktop\RP\app\demo\cook.jpg")
   # img_eng = Image.open(image_path_eng)
   #Providing the tesseract executable - location to pytesseract librar
   pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   # pytesseract.tesseract_cmd = path_to_tesseract
   custom_config = r'-l tam+sin+eng --psm 6'
   # Passing the image object to image_to_string() function -  This function will extract the text from the image
   text_eng = pytesseract.image_to_string(img_eng, config=custom_config)
    # Saving the converted text
    # exporting the result abd saved in recognized speech
   with open('recognized-english.txt',mode ='w') as file: 
    file.write("Recognized Speech:") 
    file.write("\n") 
    file.write(text_eng[:-1]) 
    print("ready english!")
   #  print(text_eng)
    #Detect orginal language
    detect_langs(text_eng)
    #  init the Google API translator
    translator = Translator()
    translation = translator.translate(text_eng)
    # translate a text to english for instance
    print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
    return text_eng




    
    