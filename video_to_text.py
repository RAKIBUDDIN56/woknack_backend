import speech_recognition as sr 
import moviepy.editor as mp


# video to audio conversion

def video_to_text(video):
    # clip = mp.VideoFileClip(r"C:\Users\user\Desktop\RP\app\demo\cooking1.mp4") 
    video.audio.write_audiofile(r"converted.wav", codec='pcm_s16le')

#  Define recognizer

    r = sr.Recognizer()

    audio = sr.AudioFile(r"converted.wav")

    with audio as source:
        audio_file = r.record(source)
    result = r.recognize_google(audio_file)

#  Export results
# exporting the result abd saved in recognized speech
    with open('recognized.txt',mode ='w') as file: 
        file.write("Recognized Speech:") 
        file.write("\n") 
        file.write(result) 
        print("ready!")
    return result    