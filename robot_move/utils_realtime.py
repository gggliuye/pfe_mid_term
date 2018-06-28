import glob
import os
import time
import urllib.request
import random
import numpy as np


ip = "192.168.1.121"
outfile = 'keys/new_image/'
port = str(8880)

def start_get_image_from_robot(name=0):
    #outfile0 = outfile + str(name) +'.jpg'
    infile = 'http://192.168.1.121:'+ port +'/image/'
    os.system('wget -r -q -nH --cut-dirs=7 --reject="index.html*" ' + "-P " + outfile + " "+ infile )

def command_camera(PWM, PUL, CHANNEL, TIME=0.01):
    command = str(PWM) +"," + str(PUL)+"," + str(CHANNEL) +"," + str(TIME)
    page = urllib.request.urlopen("http://"+ ip +":"+ port +"/move_pca.py?COMMAND=" + command)
    return 0

def command_robot(PWM, MOV, TIME, CAM, qqq ):
    command = str(PWM) +"," + str(MOV)+"," + str(TIME)+"," + str(CAM)+"," + str(qqq)
    page = urllib.request.urlopen("http://"+ ip +":"+ port +"/move.py?COMMAND=" + command)
    return 0

def take_one_photo(name=0, TIME=0.01, NUM=1):
    command = str(TIME) +"," + str(NUM) + "," + str(name)
    page = urllib.request.urlopen("http://"+ ip +":"+ port +"/camera.py?COMMAND=" + command)
    #start_get_image_from_robot()
    return 0

def scan_surrounding(n):
    # 500 to 2400
    PWM = 50
    PUL = 500
    for i in range(int(n)):
        command_camera(PWM, PUL, 0)
        take_one_photo(i)
        PUL = PUL + 1900/(n-1)
    start_get_image_from_robot()
    return 0

def scan_surrounding_small_range(n):
    # 500 to 2400
    PWM = 50
    PUL = 1000
    if n == 1:
        command_camera(50, 1450, 0)
        take_one_photo(1)
        start_get_image_from_robot()
        return 0

    for i in range(int(n)):
        command_camera(PWM, PUL, 0)
        take_one_photo(i)
        PUL = PUL + 900/(n-1)
    start_get_image_from_robot()
    return 0

def reset_servo(pil = 950):
    PWM = 50
    PUL = pil
    TIME = 0
    CHANNEL = 1
    command_camera(PWM, PUL, CHANNEL)

def reset_servo2():
    PWM = 50
    PUL = 1380
    TIME = 0
    CHANNEL = 0
    command_camera(PWM, PUL, CHANNEL)

def go_ahead_and_turn(l):
    PWM = 50
    MOV = 'fro'
    TIME = max((l*50)*0.5/14, 0)
    TIME = min(TIME, 1.5)
    CAM = 'False'
    III = 0
    # move ahead
    if TIME != 0:
        command_robot(PWM, MOV, TIME, CAM, III)
    else:
        MOV = random.choice(['lef','rig', 'lef'])
        TIME = 0.35
        command_robot(PWM, MOV, TIME, CAM, III)
    return 0
