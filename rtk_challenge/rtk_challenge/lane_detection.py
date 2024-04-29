#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
# from tf2_ros import LookupException, ConnectivityException, ExtrapolationException, Buffer
# from tf2_ros import TransformListener
from geometry_msgs.msg import Twist # velocity commandsf
from math import sqrt, cos, sin, pi, atan2, degrees, acos
import numpy as np
import sys
import cv2
import pyzed.sl as sl # Stereolab to get the image frame

# PID Controller to Move Car within track (i.e., move at straight path)
class PID:
    def __init__(self, Kp, Td, Ti, dt):
        self.Kp = Kp # proportional gain
        self.Td = Td # derivative time
        self.Ti = Ti # integral time
        self.curr_error = 0 # to compute the 
        self.prev_error = 0
        self.sum_error = 0 # accumulator for the integral term
        self.prev_error_deriv = 0
        self.curr_error_deriv = 0
        self.control = 0 # output of the PID Controller
        self.dt = dt # to compute the change in error

    def update_control(self, current_error, reset_prev=False):        
        # Update the current error based on feedback value
        self.curr_error = current_error # store the current_error
        self.sum_error += self.curr_error # accumulator
        self.curr_error_deriv = (self.curr_error - self.prev_error) / self.dt # change in error

        # Compute the Proportional, Integral and Derivative terms
        P = self.Kp * self.curr_error # Proportional term
        I = (self.Kp / self.Ti) * self.sum_error * self.dt # Integral term
        D = self.Kp * self.Td * self.curr_error_deriv # Derivative term

        #print("Integral term:", I)

        # Note: for tuning, set the Kp first; then set the Td to adjust/decay the oscillation of the overshoot; lastly the Ti value for the offset of the steady state error
        #self.control = P + D + I 
        self.control = P + D

        # Store the previous error for the next laser scan computation
        self.prev_error = self.curr_error
        pass

    def get_control(self):
        return self.control

# ZED Camera Calibration for Lane Detection
class StereoCameraCalibration:
    def __init__(self):
        # Create a ZED2 Camera object
        self.zed = sl.Camera() # sl = StereoLabs

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 opr HD1200 video mode, depending on camera type.
        init_params.camera_fps = 30  # Set fps at 30

        self.camera_frame = sl.Mat()
        
        self.runtime_parameters = sl.RuntimeParameters()

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit()
        
        self.image_size = self.zed.get_camera_information().camera_configuration.resolution
        self.car_origin = [(self.image_size.width)/2, (self.image_size.height)] # origin of the camera based of image coordinate frame

        self.grad_thresh = 0.0
        self.len_thresh = 1000
        self.y_thresh = self.image_size.height*0.5

    """--------------------------------------------------------------------------------------------------------------"""

    def calibrate_stereo_cameras(left_images, right_images, board_size, square_size):
        # Prepare object points
        objp = np.zeros((np.prod(board_size), 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
        
        # Arrays to store object points and image points from all images
        objpoints = []  # 3D points in real world space
        left_imgpoints = []  # 2D points in left image plane
        right_imgpoints = []  # 2D points in right image plane
        
        # Find corners in chessboard images
        for left_img, right_img in zip(left_images, right_images):
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            left_ret, left_corners = cv2.findChessboardCorners(left_gray, board_size, None)
            right_ret, right_corners = cv2.findChessboardCorners(right_gray, board_size, None)
            
            # If corners are found, add object points and image points
            if left_ret and right_ret:
                objpoints.append(objp)
                left_imgpoints.append(left_corners)
                right_imgpoints.append(right_corners)
        
        # Perform stereo calibration
        ret, left_mtx, left_dist, right_mtx, right_dist, R, T, E, F = cv2.stereoCalibrate(
            objpoints, left_imgpoints, right_imgpoints, None, None, None, None, (left_gray.shape[1], left_gray.shape[0])
        )
        
        # Rectify cameras
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(left_mtx, left_dist, right_mtx, right_dist, (left_gray.shape[1], left_gray.shape[0]), R, T)
        
        return left_mtx, left_dist, right_mtx, right_dist, R1, R2, P1, P2, Q
    
    def rectify_stereo_images(left_img, right_img, left_mtx, left_dist, right_mtx, right_dist, R1, R2, P1, P2):
        # Undistort and rectify images
        left_mapx, left_mapy = cv2.initUndistortRectifyMap(left_mtx, left_dist, R1, P1, (left_img.shape[1], left_img.shape[0]), cv2.CV_32FC1)
        right_mapx, right_mapy = cv2.initUndistortRectifyMap(right_mtx, right_dist, R2, P2, (right_img.shape[1], right_img.shape[0]), cv2.CV_32FC1)
        
        left_rectified = cv2.remap(left_img, left_mapx, left_mapy, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, right_mapx, right_mapy, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def compute_disparity(left_rectified, right_rectified): # compute disparity map
        # Convert images to grayscale
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity map
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(left_gray, right_gray)
        
        return disparity

    def compute_depth(self, left_img, right_img, left_mtx, left_dist, right_mtx, right_dist, Q):
        # Rectify stereo images
        left_rectified, right_rectified = self.rectify_stereo_images(left_img, right_img, left_mtx, left_dist, right_mtx, right_dist)
        
        # Compute disparity map
        disparity = self.compute_disparity(left_rectified, right_rectified)
        
        # Compute depth map
        depth_map = cv2.reprojectImageTo3D(disparity, Q)
        
        return depth_map
    
    def initialise_calibration(self):
        # Load stereo images (left and right)
        left_img = cv2.imread('left_image.jpg')
        right_img = cv2.imread('right_image.jpg')

        # Calibration board parameters
        board_size = (9, 6)  # Number of inner corners in the calibration board
        square_size = 0.02  # Size of one square in meters

        # Calibrate stereo cameras
        left_mtx, left_dist, right_mtx, right_dist, R1, R2, P1, P2, Q = self.calibrate_stereo_cameras([left_img], [right_img], board_size, square_size)

        # Rectify stereo images
        left_rectified, right_rectified = self.rectify_stereo_images(left_img, right_img, left_mtx, left_dist, right_mtx, right_dist, R1, R2, P1, P2)

        # Compute disparity map
        disparity = self.compute_disparity(left_rectified, right_rectified)

        # Display rectified images and disparity map
        # cv2.imshow('Left Rectified', left_rectified)
        # cv2.imshow('Right Rectified', right_rectified)
        # cv2.imshow('Disparity Map', disparity)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass
    
    """--------------------------------------------------------------------------------------------------------------"""

    def get_camera_frame(self): # ZED Camera
        # Capture 50 frames and stop
        i = 0
        image = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        while i < 50:
            # Grab an image, a RuntimeParameters object must be given to grab()
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # A new image is available if grab() returns SUCCESS
                left_camera_frame = self.zed.retrieve_image(image, sl.VIEW.LEFT) # Left Camera Frame
                right_camera_frame = self.zed.retrieve_image(image, sl.VIEW.RIGHT) # Right Camera Frame
                timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
                print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                    timestamp.get_milliseconds()))
                i = i + 1

        return left_camera_frame, right_camera_frame
    
    def region_of_interest(self, img, vertices):
        # Create a mask with zeros
        mask = np.zeros_like(img)
        
        # Fill the region of interest with white color (255)
        cv2.fillPoly(mask, vertices, 255)
        
        # Bitwise AND between the mask and the input image
        masked_img = cv2.bitwise_and(img, mask)
        
        return masked_img
    
    """
    def detect_lane(self, left_camera_frame, right_camera_frame): 
        left_lane = self.line_detection(left_camera_frame)
        right_lane = self.line_detection(right_camera_frame)
        return left_lane, right_lane # return left & right lane images of track with line drawn
    """
    def detect_lane(self, camera_frame):
        lane_img = self.line_detection(camera_frame)

        return lane_img
    """
    def compute_actual_bounding_point(self, left_camera_frame, right_camera_frame): # Compute the Actual Bounding Box of the Object 
        # ZED Camera; formula from the lecture notes
        baseline = 12 # metric: cm; measure the physical distance between the left camera and right camera
        origin_left_camera_position = None # left camera center
        origin_right_camera_position = None # right camera center

        focal_length = None

        origin_left_frame = [x_left, y_left]
        origin_right_frame = [x_right, y_right]
        disparity = origin_left_frame[0] - origin_right_frame[0] # metric

        absolute_depth_of_point = (focal_length * baseline) / disparity
        return absolute_depth_of_point
    """

    def get_intersection(self, line):

        #takes in two numpy arrays

        grad = (line[2]-line[0])/(line[3]-line[1]+1e-6)

        x_inter = line[0]+grad*(self.y_thresh-line[1])

        return x_inter

    def compute_lane_deviation(self): # Parallel Stereo Cameras; Epipolar Geometry
        # ZED Camera
        # TODO: find out the x-y image coordinates of the center_line
        # Boundary lines of the track as imaginary walls
        # left_camera_frame, right_camera_frame = self.get_camera_frame()
        camera_frame = self.get_webcam_frame()
        lanes = self.detect_lane(camera_frame) # limits of the track; left & right lane image
        
        # print("lanes", lanes)

        x_coordinates = []

        for lane in lanes:
            x = self.get_intersection(lane[0])
            if x>0 and x< self.image_size.width:
                x_coordinates.append(x)
        # print("x", x_coordinates)

        if len(x_coordinates):

            x_coordinates = np.array(x_coordinates)
            x_coordinates_sorted = np.sort(x_coordinates)

            # print(x_coordinates, x_coordinates_sorted)

            # print("Width", self.image_size.width/2)
            left_bounding_pixel = x_coordinates_sorted[0]
            right_bounding_pixel = x_coordinates_sorted[-1]
        
        else:

            left_bounding_pixel = self.image_size.width/2
            right_bounding_pixel = self.image_size.width/2


        #right_bounding_pixel = np.min(np.where(x_coordinates>self.image_size.width/2, x_coordinates, np.inf)) # left limit pixel on the x-axis
        #left_bounding_pixel = np.max(np.where(x_coordinates<self.image_size.width/2, x_coordinates,-np.inf)) # right limit pixel on the x-axis 
        #bounding_corner_left = self.compute_actual_bounding_point(left_camera_frame, right_camera_frame) # x-coordinate of left corner
        #bounding_corner_right = self.compute_actual_bounding_point(left_camera_frame, right_camera_frame) # x-coordinate of right corner
        
        # print(f"left: {left_bounding_pixel}, right: {right_bounding_pixel}")

        # self.display_image()
        
        center_point = (left_bounding_pixel + right_bounding_pixel) / 2 # centerline of the track
        offset = center_point - self.car_origin[0]
        
        # offset_centerline = self.car_origin[0] - center_point # positive_offset = steer left; negative_offset = steer right
        # offset_right_limit = right_bounding_pixel - self.car_origin[0]
        # offset_left_limit = self.car_origin[0] - left_bounding_pixel

        return offset
       
    def get_webcam_frame(self): # WebCam
        # Capture
        #ret, camera_frame = self.vid.read()
        
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        
            self.zed.retrieve_image(self.camera_frame, sl.VIEW.LEFT)       
        
        # cv2.imshow("Camera View",  camera_frame.get_data())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        #Returns grayscale frame
        return cv2.cvtColor(self.camera_frame.get_data(), cv2.COLOR_RGBA2GRAY)

        #camera_frame = cv2.cvtColor(camera_frame, cv2.RGB2BGR)
    
    def line_detection(self, camera_frame): # draw line on lane through hough transform; cv2
        # Convert image to grayscale
        # gray = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(camera_frame, (5, 5), 0)
        
        # Perform edge detection using Canny
        edges = cv2.Canny(blurred, 50, 100, apertureSize=3)
   
        
        vertices = np.array([[(0,self.image_size.height*0.55),(self.image_size.width, self.image_size.height*0.55),(self.image_size.width,self.image_size.height),(0,self.image_size.height)]], dtype=np.int32)
        mask = np.zeros_like(edges)
        #Mask canny edge map
        cv2.fillPoly(mask, vertices, 255)
        edges = cv2.bitwise_and(edges, mask)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=50) 
        #lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        # edges = cv2.resize(edges, (640, 480))
        # cv2.imshow('edges detected',edges)
        # cv2.waitKey(1)

        # new_lines = []

        #line filtering
        # if lines is not None:
        #     for line in lines:
        #         gradient = (line[0][3]-line[0][1])/(line[0][2]-line[0][0]+1e-6)
        #         length = (line[0][3]-line[0][1])**2+(line[0][2]-line[0][0])**2
        #         if abs(gradient) > self.grad_thresh and length > self.len_thresh:
        #             new_lines.append(line)

        #             # x1, y1, x2, y2 = line[0] # list within a list
        #             # cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        return lines
        """
        lines = [
            [[x1, y1, x2, y2]],  # Line 1
            [[x1, y1, x2, y2]],  # Line 2
            # ... more lines
        ]
        each element in the 'lines' list is a list itself, containing the coordinates '[x1,y1,x2,y2]' of the endpoints of the line
        starting point '(x1,y1)' and ending point '(x2,y2)'
        
        #print(str(len(lines))) # x1,y1; x2,y2
        #print(lines) 
        #print(lines.shape[0]) # numpy array
        """

class BuggyCar(Node):
    def __init__(self):
        super().__init__('lane_detection')

        timer_period = 0.1
        cmd_timer_period = 0.02

        #self.forward_speed = float(self.declare_parameter("forward_speed").value) # 0.8m/s
        self.forward_speed = 2.0 # 0.8m/s
        #self.desired_distance_from_wall = float(self.declare_parameter("desired_distance_from_wall").value) # 1m: setpoint
        self.desired_distance_from_lane = 0.2 # width of lane: 1.2m; 0.2m from the boundary lane        
        self.desired_error_from_centerline = 0.1
        self.hz = int(1/timer_period) # max frequency of the zed camera frame publishing rate; find at the zed2 yaml file

        # TODO: set up the command publisher to publish at topic '/cmd_vel'
        # using geometry_msgs/Twist messages
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', QoSProfile(depth=10)) # buggy car cmd_vel topic
        
        # self.timer = self.create_timer(timer_period, self.lane_detection_callback)

        self.cmd_timer = self.create_timer(cmd_timer_period, self.cmd_vel_callback)
        
        print("Node started")
        
        #Camera Calibration
        # self.stereo_camera = StereoCameraCalibration()
        # stereo_camera.display_image()

        # TODO: set up the object detection subscriber
        # this will set up a callback function that gets executed
        # upon each spin() call, as long as a object stamped
        # message has been published in the meantime by another node

        # Initialise PID controller
        self.Kp = self.declare_parameter('Kp',0.2).value
        self.Td = self.declare_parameter('Td',0.05).value
        self.Ti = self.declare_parameter('Ti',100).value
        #self.vel_controller = PID(0.8,0.075,1,1/self.hz) # initialise the parameters for the PID controller
        # p-gain (can work): 0.75, 0.75, 0.8
        # d-gain (can work): 0.05, 0.075, 0.05
        # d-gain (cannot work): 0.1

        #self.tf_buffer = Buffer()
        #self.tf_listener = TransformListener(self.tf_buffer, self)

        self.count = 1 # counter for each image frame

        self.cmd = Twist() # velocity_command
        self.cmd.linear.x = self.forward_speed
        self.cmd.angular.z = 0.015

    def update_parameters(self): # Tune PID parameters from rqt_configure
        self.Kp = self.get_parameter('Kp').value
        self.Td = self.get_parameter('Td').value
        self.Ti = self.get_parameter('Ti').value
      
        # self.get_logger().info('Kp: %s' % self.Kp)
        # self.get_logger().info('Td: %s' % self.Td)
        # self.get_logger().info('Ti: %s' % self.Ti)
        return 
    
    def cmd_vel_callback(self):

        # print("Publishing speed")

        self.cmd_pub.publish(self.cmd)

    def lane_detection_callback(self): # MAIN PROGRAM: line detection of center line between boundary lanes
        """ 1) Lane Detection and Compute the Lane Deviation """
        
        offset = self.stereo_camera.compute_lane_deviation()
        offset_metres = offset *0.001

        print("Offset", offset)
        # print("Offsets", center_offset, left_offset, right_offset)

        # self.stereo_camera.display_image()
        
        lane_error = offset_metres  # deviation from the center line of the lane in the x-axis; center_offset

        """ 2) Execute PID Controller """
        # 2.1) Get updated Kp, Td, Ti values in rqt_reconfigure (Dynamic Reconfigure)
        self.update_parameters() # update the PID values to the controller
        vel_controller = PID(self.Kp,self.Td,self.Ti,1/self.hz) # initialise the parameters for the PID controller

        # 2.2) Update the PID Controller
        #vel_controller.update_control(lane_error)
        #current_vel_output = vel_controller.get_control()
        #print("current_vel_output: ",current_vel_output)

        vel_controller.update_control(lane_error)
        vel_output = vel_controller.get_control() # update the output of the PID Controller based on the current error
        #print("projected_vel_output: ",projected_vel_output)
        
        """ 3) Publish the velocity command based from the Output of the PID Controller; may need to remap to the max speed of the robot """
        
        print("Forward Speed: ", self.forward_speed)
        self.cmd.angular.z = vel_output # signs of the value should match the steering direction  
        
        if(vel_output >0):
            print("Turning Right: ", vel_output)

        if(vel_output<0):
            print("Turning Left: ", vel_output)
        # self.cmd_pub.publish(self.cmd) # publish velocity commands to buggy car


def main(args=None):
    rclpy.init(args=args)

    buggy_car = BuggyCar()

    rclpy.spin(buggy_car)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    buggy_car.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
