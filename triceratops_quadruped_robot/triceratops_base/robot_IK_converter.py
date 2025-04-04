#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys, os
sys.path.append(os.path.expanduser("~/UnityRos2_ws/src/unity_robotics_demo_msgs"))
from unity_robotics_demo_msgs.msg import JointState
import numpy as np
from datetime import datetime

def transform_to_fr(x, y, z, leg_type):
    if leg_type == 'fr':
        return x, y, z
    elif leg_type == 'fl':
        return x, -y, z
    elif leg_type == 'hr':
        return -x, y, z
    elif leg_type == 'hl':
        return -x, -y, z
    else:
        raise ValueError(f"Unknown leg type: {leg_type}")

def inverse_transform_angles(gamma, alpha, beta, leg_type):
    if leg_type == 'fr':
        return gamma, alpha, beta
    elif leg_type == 'fl':
        return -gamma, alpha, beta
    elif leg_type == 'hr':
        return gamma, -alpha, -beta
    elif leg_type == 'hl':
        return -gamma, -alpha, -beta
    else:
        raise ValueError(f"Unknown leg type: {leg_type}")

def inverse_kinematics(x, y, z):
    # h = 0.0365
    h = 0.0375
    d = (y ** 2 + z ** 2) ** 0.5
    l = (d ** 2 - h ** 2) ** 0.5
    gamma1 = - np.arctan(h / l)
    gamma2 = - np.arctan(y / z)
    gamma = gamma2 - gamma1

    s = (l ** 2 + x ** 2) ** 0.5
    # hu = 0.065
    # hl = 0.065
    hu = 0.13
    hl = 0.13
    n = (s ** 2 - hl ** 2 - hu ** 2) / (2 * hu)
    beta = -np.arccos(n / hl)

    alpha1 = - np.arctan(x / l)
    alpha2 = np.arccos((hu + n) / s)
    alpha = alpha2 + alpha1

    return np.array([gamma, alpha, beta])

class JointStateRecorder(Node):
    def __init__(self):
        super().__init__('joint_state_recorder')
        
        # Create the subscriber
        self.subscription = self.create_subscription(
            JointState,
            'joint_states_play',
            self.joint_callback,
            10)
        
        # Initialize storage for frames, endpoints, and timestamps
        self.frames = []
        self.endpoints = []
        self.timestamps = []
        self.recording = True
        self.first_frame = None
        self.min_frames = 20  # Minimum number of frames before checking for repetition
        
        # Store leg origin bias values
        self.leg_origins = {
            'fr': (0.128, -0.055, 0),
            'fl': (0.128, 0.055, 0),
            'hr': (-0.128, -0.055, 0),
            'hl': (-0.128, 0.055, 0)
        }
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.angles_file = f'joint_angles_{timestamp}.txt'
        self.endpoints_file = f'leg_endpoints_{timestamp}.txt'
        
        self.get_logger().info('Joint State Recorder has been started')

    def detect_pattern_completion(self, current_frame, threshold=1e-2):
        """
        Detect if the motion pattern is complete by comparing with first frame
        """
        if self.first_frame is None or len(self.frames) < self.min_frames:
            return False

        # Compare current frame with first frame
        return np.all(np.abs(current_frame - self.first_frame) < threshold)

    def process_frame(self, msg):
        """Process a single frame of joint states into angles and endpoints."""
        leg_order = ['fr', 'fl', 'hr', 'hl']
        frame_angles = []
        frame_endpoints = []
        
        # Process each leg
        for i, leg in enumerate(leg_order):
            # Get raw coordinates from message
            y, z, x = -msg.x[i], msg.y[i], msg.z[i]
            
            # Store the original endpoint coordinates with origin bias
            origin_x, origin_y, origin_z = self.leg_origins[leg]
            endpoint = [x + origin_x, y + origin_y, z + origin_z]
            frame_endpoints.extend(endpoint)
            
            # Transform coordinates for inverse kinematics
            x_fr, y_fr, z_fr = transform_to_fr(x, y, z, leg)
            
            # Calculate angles
            gamma, alpha, beta = inverse_kinematics(x_fr, y_fr, z_fr)
            
            # Transform angles back
            gamma, alpha, beta = inverse_transform_angles(gamma, alpha, beta, leg)
            
            # Add to frame angles
            frame_angles.extend([gamma, alpha, beta])
        
        return np.array(frame_angles), np.array(frame_endpoints)

    def save_data(self):
        """Save recorded frames and endpoints to files."""
        if not self.frames:
            return
        
        # Save joint angles
        frames_array = np.array(self.frames)
        timestamps_array = np.array(self.timestamps).reshape(-1, 1)
        angles_output = np.hstack((frames_array, timestamps_array))
        
        np.savetxt(
            self.angles_file, 
            angles_output,
            fmt='%.6f',
            header=f'Joint angles matrix: {frames_array.shape[0]} frames, 12 joints per frame + timestamp\n'
                  'Order: fr[gamma,alpha,beta] fl[gamma,alpha,beta] hr[gamma,alpha,beta] hl[gamma,alpha,beta] timestamp'
        )
        
        # Save endpoints
        endpoints_array = np.array(self.endpoints)
        endpoints_output = np.hstack((endpoints_array, timestamps_array))
        
        np.savetxt(
            self.endpoints_file, 
            endpoints_output,
            fmt='%.6f',
            header=f'Leg endpoints matrix: {endpoints_array.shape[0]} frames, 12 coordinates per frame + timestamp\n'
                  'Order: fr[x,y,z] fl[x,y,z] hr[x,y,z] hl[x,y,z] timestamp'
        )
        
        self.get_logger().info(f'Saved {len(self.frames)} frames to {self.angles_file} and {self.endpoints_file}')

    def joint_callback(self, msg):
        """Process incoming joint states and record angles and endpoints."""
        try:
            if not self.recording:
                return
            
            # Process current frame
            current_angles, current_endpoints = self.process_frame(msg)
            current_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds
            
            # Store first frame for comparison
            if self.first_frame is None:
                self.first_frame = current_angles
            
            # Store frame, endpoints, and timestamp
            self.frames.append(current_angles)
            self.endpoints.append(current_endpoints)
            self.timestamps.append(current_time)
            
            # Check for pattern completion
            if self.detect_pattern_completion(current_angles):
                self.recording = False
                self.save_data()
                self.get_logger().info(f'Pattern completed after {len(self.frames)} frames')
                return
            
            # Log progress
            if len(self.frames) % 10 == 0:
                self.get_logger().info(f'Recorded {len(self.frames)} frames')
            
        except Exception as e:
            self.get_logger().error(f'Error processing joint states: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    
    joint_recorder = JointStateRecorder()
    
    try:
        rclpy.spin(joint_recorder)
    except KeyboardInterrupt:
        joint_recorder.save_data()
    finally:
        joint_recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()