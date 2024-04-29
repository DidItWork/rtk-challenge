#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import tf2_ros
# import tf2_geometry_msgs
#from tf2_ros import LookupException, ConnectivityException, ExtrapolationException, Buffer
#from tf2_ros import TransformListener
from rclpy.node import Node
from zed_interfaces.msg import ObjectsStamped
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose
from tf2_geometry_msgs import do_transform_pose
import numpy as np
import sys
import math
import csv

DETECTION_DISTANCE = 1.5
object_positions = np.array([[1.5,0,0], [6.0,0,0], [10.0,0,0], [14.0,0,0]])
object_labels = ["","","",""]

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z # in radians

def rotation3d(roll: float, pitch: float, yaw: float, position):
        position_np = np.array(position)

        #yaw => z axis roation,roll=> x axis rotation, pitch => y axis rotation

        yaw_rotation = np.matrix([[ math.cos(yaw), -math.sin(yaw), 0 ],
                   [ math.sin(yaw), math.cos(yaw) , 0 ],
                   [ 0    , 0    , 1 ]])
        roll_rotation = np.matrix([[ 1   , 0  , 0 ],
                    [ 0, math.cos(roll), -math.sin(roll)],
                   [ 0, math.sin(roll), math.cos(roll) ]]) 
        pitch_rotation = np.matrix([[ math.cos(pitch), 0, math.sin(pitch)],
                                    [ 0 , 1 , 0 ],
                   [ -math.sin(pitch), 0, math.cos(pitch) ]]) 
        return  position_np * yaw_rotation * roll_rotation * pitch_rotation
    


def calculatePostiion(object_pos, odom_position, odom_rotation):
    #rotate object pos first
    rotated_object_pos = rotation3d(odom_rotation[0], odom_rotation[1], odom_rotation[2], object_pos)
    #translate
    translated_object_pos = rotated_object_pos + np.array(odom_position)
    return translated_object_pos.tolist()[0]

def calculatedistance(pos1, pos2):
    #only calcultate distance within 2d plane
    # return math.sqrt((pos1[0]-pos2[0]) ** 2 + (pos1[1]-pos2[1]) ** 2)
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])
    

class ObjectMapping(Node):
    def __init__(self):
        super().__init__('object_mapping')

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.subscription_object_detection = self.create_subscription(
            ObjectsStamped,
            '/zed/zed_node/obj_det/objects',
            self.object_detection_callback,
            10)
        
#         self.subscription_odom = self.create_subscription(
#             geometry_msgs.msg.PoseStamped,
# # https://answers.ros.org/question/389795/difference-in-the-out-of-rtabmaplocalization_pose-and-tf_echo-base_link-map/
#             '/rtabmap/localization_pose',
#             self.odom_update_callback,
#             10)
        self.subscription_object_detection  # prevent unused variable warning
        # self.subscription_odom

        self.odom_position = []
        self.odom_oriendtation = []
        self.odom_rotations = []
        self.labeled_objects = []
        #open csv file before destructor
        # self.fh = open('object_positions.csv', 'w', newline='')
    
        print("Initialized")

    def object_detection_callback(self, msg):

        print("objects detected")

        for obj in msg.objects:

            # if calculatedistance([0,0,0], obj.position) > DETECTION_DISTANCE:
            #     continue

            print(obj.label, obj.position)

            object_pose = Pose()

            object_pose.position.x = float(obj.position[0])
            object_pose.position.y = float(obj.position[1])
            object_pose.position.z = float(obj.position[2])
            
            object_pose.orientation.x = 0.
            object_pose.orientation.y = 0.
            object_pose.orientation.z = 0.
            object_pose.orientation.w = 1.0

            # object_pose.header.frame_id = "zed_left_camera_frame"
            # object_pose.header.stamp = self.get_clock().now().to_msg()

            #get detected obj's position
            # obj_detected_pos = [obj.position.x, obj.position.y, obj.position.z]
            # calculated_pos = calculatePostiion(obj_detected_pos, self.odom_position, self.odom_rotations)
            # object_pose = self.tf_buffer.transform(object_pose, "odom")


                
            # print(object_pose.header)

            # transform = self.tf_buffer.lookup_transform("map",
            #                                             "zed_left_camera_frame",
            #                                             rclpy.time.Time())  
            
            #Transform pose to map frame
            # object_pose = tf2_geometry_msgs.do_transform_pose(object_pose, transform)

                
            transform_map = self.tf_buffer.lookup_transform("odom", "zed_left_camera_frame", rclpy.time.Time().to_msg())            

            object_pose = do_transform_pose(object_pose, transform_map)

            calculated_pos = np.array([object_pose.position.x, object_pose.position.y, object_pose.position.z])
            # calculated_pos[0] = object_pose.pose.position.x
            # calculated_pos[1] = object_pose.pose.position.y
            # calculated_pos[2] = object_pose.pose.position.z

            distances = np.sum(np.abs(object_positions-calculated_pos),axis=1)

            selected_object = np.argmin(distances)

            object_positions[selected_object] = 0.8 * object_positions[selected_object] + 0.2 * calculated_pos

            object_labels[selected_object] = obj.label     

            # print("cannot obtain transform") 

    def odom_update_callback(self, msg):
        #check whether this message works well or not
        self.odom_position= [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        #check whether this message works well or not
        self.odom_oriendtation = [msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        #save car's rotation to euler angle
        self.odom_rotations = [euler_from_quaternion(self.odom_oriendtation)]


def main(args=None):
    rclpy.init(args=args)

    obj_map = ObjectMapping()

    try:
        rclpy.spin(obj_map)
    except KeyboardInterrupt:
        print(object_positions)
        with open('object_positions.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "x", "y", "z"])
            for label, obj_pos in zip(object_labels, object_positions):
                writer.writerow([label]+list(obj_pos))
        print ('My application is ending!')

    obj_map.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
