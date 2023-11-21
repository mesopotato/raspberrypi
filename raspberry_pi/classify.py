# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run image classification."""

import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import asyncio
from telegram import Bot, InputFile
import datetime 

#Telegram bot configuration 
BOT_TOKEN = "6961444963:AAGOpXl9mHmrQD7ykt5zuCS1ixi4J_4YoFI"
CHAT_ID = "65654377"
bot = Bot(token=BOT_TOKEN)
cadence = 6

async def send_cat_photo(frame):
  try:
    print("sending to telegram")
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S_%f") 
    photo_name = f'{formatted_time}_cat_detected.jpg'
    photo_path=f'/home/yannick/fileServer/cats/{photo_name}'
    print(photo_path)
    cv2.imwrite(photo_path, frame)
    temp_photo = 'cat_detected.jpg'
    cv2.imwrite(temp_photo, frame)
    with open(temp_photo, 'rb') as photo:
      await bot.send_photo(chat_id=CHAT_ID, photo=InputFile(photo))
  except TimedOut as e: 
    print(f"Failed to send the photo to telegram due to timeout : {e}")
  except Exception as e: 
    print(f"an error occured while sending the photo to telegram or while storing the photo : {e}")   
  #cv2.imwrite(photo_path, frame)
  #with open(photo_path, 'rb') as photo:
  #  await bot.send_photo(chat_id=CHAT_ID, photo=InputFile(photo))

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10


async def run(model: str, max_results: int, score_threshold: float, num_threads: int,
        enable_edgetpu: bool, camera_id: int, width: int, height: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
      model: Name of the TFLite image classification model.
      max_results: Max of classification results.
      score_threshold: The score threshold of classification results.
      num_threads: Number of CPU threads to run the model.
      enable_edgetpu: Whether to run the model on EdgeTPU.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
  """

  # Initialize the image classification model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)

  # Enable Coral by this setting
  classification_options = processor.ClassificationOptions(
      max_results=max_results, score_threshold=score_threshold)
  options = vision.ImageClassifierOptions(
      base_options=base_options, classification_options=classification_options)

  classifier = vision.ImageClassifier.create_from_options(options)

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  limit = 0
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create TensorImage from the RGB image
    tensor_image = vision.TensorImage.create_from_array(rgb_image)
    # List classification results
    categories = classifier.classify(tensor_image)

    cat_detected = False 

    # Show classification results on the image
    for idx, category in enumerate(categories.classifications[0].categories):
      category_name = category.category_name
      
      if "cat" in category_name.lower():
        limit = limit + 1 
        cat_detected = True 
        print("Cat detected!")
        
      #score = round(category.score, 2)
      #result_text = category_name + ' (' + str(score) + ')'
      #text_location = (_LEFT_MARGIN, (idx + 2) * _ROW_SIZE)
      #cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
      #            _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    
    if cat_detected :
      #text = "CAT DETECTED" 
      #text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
      #text_x = (image.shape[1] - text_size[0]) // 2 
      #text_y = (image.shape[0] - text_size[1]) // 2 
      #cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      if limit >= cadence : 
          await send_cat_photo(image)
          limit = 0
    # Calculate the FPS
    if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
      end_time = time.time()
      fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    #fps_text = 'FPS = ' + str(int(fps))
    #text_location = (_LEFT_MARGIN, _ROW_SIZE)
    #cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
    #            _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    #cv2.imshow('image_classification', image)

  cap.release()
  cv2.destroyAllWindows()


#async def main():
async def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of image classification model.',
      required=False,
      default='efficientnet_lite0.tflite')
  parser.add_argument(
      '--maxResults',
      help='Max of classification results.',
      required=False,
      default=3)
  parser.add_argument(
      '--scoreThreshold',
      help='The score threshold of classification results.',
      required=False,
      type=float,
      default=0.0)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  await run(args.model, int(args.maxResults),
      args.scoreThreshold, int(args.numThreads), bool(args.enableEdgeTPU),
      int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  #main()
   loop = asyncio.get_event_loop()
   loop.run_until_complete(main())   
