#!/usr/bin/python3
# -*- coding: utf-8 -*

import cgi
import time
from util_robot import *
from picamera import PiCamera

#get fields from request
form=cgi.FieldStorage()
COMMAND=form.getvalue("COMMAND")
my_list = COMMAND.split(",")


PWMString = my_list[0]
MOVString = my_list[1]
TIMEString = my_list[2]
CAM = my_list[3]

PWM = float(PWMString)
TIME = float(TIMEString)

Robot=AlphaBot2()
if CAM == 'True':
    camera = PiCamera()
    move_robot_with_camera(Robot,camera, PWM, MOVString, TIME)
else:
    move_robot(Robot, PWM, MOVString, TIME)
