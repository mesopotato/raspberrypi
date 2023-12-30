import cv2
import asyncio
from telegram import Bot, InputFile
import datetime
import time

#Telegram bot configuration 
BOT_TOKEN = "6961444963:AAGOpXl9mHmrQD7ykt5zuCS1ixi4J_4YoFI"
CHAT_ID = "-1002044540734"
bot = Bot(token=BOT_TOKEN)

async def send_cat_photo(frame):
  try:
    print("sending to telegram")
    temp_photo = 'cat_detected.jpg'
    cv2.imwrite(temp_photo, frame)
    with open(temp_photo, 'rb') as photo:
      await bot.send_photo(chat_id=CHAT_ID, photo=InputFile(photo))
  except Exception as e: 
    print(f"an error occured while sending the photo to telegram or while storing the photo : {e}")   

def store_cat_photo(frame):
    #print("storing photo")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S_%f") 
    photo_name = f'{formatted_time}_cat_detected.jpg'
    photo_path=f'/home/yannick/fileServer/cats/{photo_name}' 
    #photo_path=f'Z:/cats/{photo_name}'
    cv2.imwrite(photo_path, frame)

def detect_cat_faces(image):
    # Load the pre-trained classifier
    #cat_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
    cat_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')

    # Convert the image to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect cat faces
    cat_faces = cat_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

    # Draw rectangles around detected cat faces
    #for (x, y, w, h) in cat_faces:
    #    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return len(cat_faces) > 0, image

def main_loop():
    cap = cv2.VideoCapture(0)
    detection_count = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detected, frame_with_detections = detect_cat_faces(frame)
        #cv2.imshow('Cat Face Detection', frame_with_detections)

        if detected:

            store_cat_photo(frame)

            detection_count += 1
            #print(f"Detection count: {detection_count}")

            if detection_count >= 6 and (time.time() - start_time) <= 60:
                asyncio.run(send_cat_photo(frame))
                detection_count = 0  # Reset count after sending photo
                start_time = time.time()  # Reset timer

        if time.time() - start_time > 60:
            detection_count = 0  # Reset count after 1 minute
            start_time = time.time()  # Reset timer

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
