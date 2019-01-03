import rospy
from std_msgs.msg import Float32
import cv2
import math
from time import sleep

class car_control:
    def __init__(self):
        self.speed_pub = rospy.Publisher("Team1_speed", Float32, queue_size=1)
        self.steerAngle_pub = rospy.Publisher("Team1_steerAngle", Float32, queue_size=1)
        rospy.init_node('cds', anonymous = True)
        rospy.Rate(10)

    def control(self, left_fit, right_fit, signWidth):
        if not rospy.is_shutdown():
            steerAngle = self.cal_steerAngle(left_fit, right_fit, signWidth)
            if math.fabs(steerAngle) >= 10 or signWidth != 0:
                self.speed_pub.publish(30)
            else:
                self.speed_pub.publish(50)


    def cal_steerAngle(self, left_fit, right_fit, signWidth):
        carPos_x = 160
        carPos_y = 240

        # Middle pos between two side of lane
        middlePos_x, middlePos_y = self.find_middlePos(left_fit, right_fit)

        if signWidth == 25 :
            print("PREPARE")
            sleep(0.5)
            for i in range(2000):
                print("RIGHT")
                steerAngle = 20
                self.steerAngle_pub.publish(steerAngle)

        elif signWidth < -25:
            for i in range(100):
                print("PREPARE")
            for i in range(100):
                print("LEFT")
                steerAngle = -20
                self.steerAngle_pub.publish(steerAngle)


        else:
            # print("STRAIGHT")
            # Can't detect lane
            if middlePos_x == -1:
                steerAngle = 0
            else:
                # Distance between MiddlePos and CarPos
                distance_x = middlePos_x - carPos_x
                distance_y = carPos_y - middlePos_y

                # Angle to middle position
                steerAngle = math.atan(distance_x / distance_y) * 180 / math.pi
                # print(middlePos_x, steerAngle)
            self.steerAngle_pub.publish(steerAngle)

        return steerAngle

    def find_middlePos(self, left_fit, right_fit):
        middlePos_y = 120
        laneWidth = 175

        # Detect nothing
        if left_fit[0] == 0 and right_fit[0] == 0:
            return -1, -1
        # Detect right side
        elif left_fit[0] == 0 and right_fit[0] != 0:
            rightSide_x = right_fit[0] * middlePos_y ** 2 + right_fit[1] * middlePos_y + right_fit[2]
            middlePos_x = rightSide_x - laneWidth/2
            return middlePos_x, middlePos_y
        # Detect left side
        elif left_fit[0] != 0 and right_fit[0] == 0:
            leftSide_x = left_fit[0] * middlePos_y ** 2 + left_fit[1] * middlePos_y + left_fit[2]
            middlePos_x = leftSide_x + laneWidth/2
            return middlePos_x, middlePos_y
        # Detect both
        else:
            leftSide_x = left_fit[0] * middlePos_y ** 2 + left_fit[1] * middlePos_y + left_fit[2]
            rightSide_x = right_fit[0] * middlePos_y ** 2 + right_fit[1] * middlePos_y + right_fit[2]
            # print("WIDTH ", rightSide_x - leftSide_x)
            middlePos_x = (leftSide_x + rightSide_x) / 2
            return middlePos_x, middlePos_y

