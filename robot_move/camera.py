#!/usr/bin/python3
# -*- coding: utf-8 -*

import cgi
import time
from AlphaBot2 import AlphaBot2
from picamera import PiCamera
from util_robot import*

form=cgi.FieldStorage()
COMMAND=form.getvalue("COMMAND")
my_list = COMMAND.split(",")

TIME = float(my_list[0])
num = int(my_list[1])

camera = PiCamera()
save_image(camera, TIME, num)
