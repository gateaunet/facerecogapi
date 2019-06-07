import os
import sys


def onWebcam():
    os.system('mjpg_streamer -i "input_uvc.so" -o "output_http.so -p 8090 -w /usr/local/share/mjpg-streamer/www/"') # start mjpgstreamer



