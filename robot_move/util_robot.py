import time
from AlphaBot2 import AlphaBot2
from PCA9685 import PCA9685
from picamera import PiCamera
import RPi.GPIO as GPIO
import glob
import os



def move_robot(Robot, PWM, MOVString, TIME, secu=True):
    Robot.setPWMA(PWM)
    Robot.setPWMB(PWM)

    if secu :
        DR = 16
        DL = 19
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(DR,GPIO.IN,GPIO.PUD_UP)
        GPIO.setup(DL,GPIO.IN,GPIO.PUD_UP)

    if (MOVString=='fro' ):
        if secu:
            start = time.clock()
            print(start)
            while (time.clock() - start < TIME ):
                DR_status = GPIO.input(DR)
                DL_status = GPIO.input(DL)
                print(DR_status, DL_status)
                if((DL_status == 0) or (DR_status == 0)):
                    Robot.stop()
                    print('get sth')
                else:
                    Robot.forward()
            Robot.stop()
        else:
            Robot.forward()
            time.sleep(TIME)
    elif (MOVString=='bac'):
        Robot.backward()
        time.sleep(TIME)
    elif (MOVString=='lef'):
        Robot.left()
        time.sleep(TIME)
    elif (MOVString=='rig'):
        Robot.right()
        time.sleep(TIME)
    else:
        Robot.stop()
    return 0

def sleep_and_take_photo(TIME,camera):
    num = int(TIME/0.5)
    for i in range(num):
        time.sleep(0.5)
        camera.capture('/home/pi/ideaversal/image/%s.jpg' % i)
    return 0

def move_robot_with_camera(Robot,camera, PWM, MOVString, TIME):
    Robot.setPWMA(PWM)
    Robot.setPWMB(PWM)
    camera.start_preview()
    if (MOVString=='fro' ):
        Robot.forward()
        sleep_and_take_photo(TIME,camera)
    elif (MOVString=='bac'):
        Robot.backward()
        sleep_and_take_photo(TIME,camera)
    elif (MOVString=='lef'):
        Robot.left()
        sleep_and_take_photo(TIME,camera)
    elif (MOVString=='rig'):
        Robot.right()
        sleep_and_take_photo(TIME,camera)
    else:
        Robot.stop()
    camera.stop_preview()
    return 0


def save_image(camera, TIME, num,name='0'):
    camera.start_preview()
    if(num == 1):
        TIME = 0
        #images = glob.glob('/home/pi/ideaversal/image/*.jpg')
        #for file in images:
            #os.remove(file)
    for i in range(num):
        time.sleep(TIME)
        imagefile = '/home/pi/ideaversal/image/' + name + '_' + str(i) +'.jpg'
        camera.capture(imagefile)
    camera.stop_preview()
    return 0
