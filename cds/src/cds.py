#!/usr/bin/env python
from __future__ import print_function
import roslib
roslib.load_manifest('cds')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:
    def __init__(self):
        #self.image_pub = rospy.Publisher("image_topic_2",Image)

        self.bridge = CvBridge()
        print("beforecallback")
        self.image_sub = rospy.Subscriber("Team1_image", Image, self.callback)

    def callback(self, data):
        print("callback")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow("Image window", cv_image)
        except CvBridgeError as e:
            print(e)

        #(rows,cols,channels) = cv_image.shape
        #if cols > 60 and rows > 60 :
        #   cv2.circle(cv_image, (50,50), 10, 255)


        cv2.waitKey(30)

        #try:
        #    self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        #except CvBridgeError as e:
        #    print(e)


if __name__ == '__main__':
    rospy.init_node('cds', anonymous=True)
    ic = image_converter()
    rospy.spin()
    # cv2.destroyAllWindows()
