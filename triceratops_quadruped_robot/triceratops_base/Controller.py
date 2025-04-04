import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from Triceratops_Config import RobotConfiguration, RobotDynamixel

from DXL_motor_control import DXL_Communication
from Gait_controller import GaitController
from Swing_controller import SwingController
from Stance_controller import StanceController

from Triceratops_IK import InverseKinematics
from Triceratops_FK import ForwardKinematics

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys, os
sys.path.append(os.path.expanduser("~/UnityRos2_ws/src/unity_robotics_demo_msgs"))
from unity_robotics_demo_msgs.msg import JointState
from unity_robotics_demo_msgs.srv import UnityAnimateService
from std_msgs.msg import String

from math import pi
import threading
import queue
import traceback
import atexit
import time

import threading
import concurrent.futures
import numpy as np

DEGREE_TO_SERVO = 4095/360

class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__('cmd_vel_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.listener_callback,
            10
        )

        self.linear_x = 0.0
        self.linear_y = 0.0
        self.linear_z = 0.0

        self.angular_x = 0.0
        self.angular_y = 0.0
        self.angular_z = 0.0

        self.get_logger().info('Vel Subscriber has been started')

    def listener_callback(self, msg):
        self.linear_x = msg.linear.x
        self.linear_y = msg.linear.y
        self.linear_z = msg.linear.z

        self.angular_x = msg.angular.x
        self.angular_y = msg.angular.y
        self.angular_z = msg.angular.z


class JointStatesPublisher(Node):
    def __init__(self):
        super().__init__('joint_states_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_states_record', 10)

        # self.joint_names = []
        # self.foot_endpoints = np.zeros([3, 4])
        # self.lumbar_angles = np.zeros(2)

        self.get_logger().info('JointStates Publisher has been started')

class JointStatesSubscriber(Node):
    def __init__(self):
        super().__init__('joint_states_subscriber')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states_play',
            self.listener_callback,
            10
        )
        # self.publisher_ = self.create_publisher(JointState, 'joint_states_play', 10)

        self.joint_names = []
        self.foot_endpoints = np.zeros([3, 4])
        self.lumbar_angles = np.zeros(2)
        self.head_angles = np.zeros(1)

        self.get_logger().info('JointStates Subscriber has been started')

    def listener_callback(self, msg: JointState):
        self.joint_names = msg.name
        self.foot_endpoints[:3, :] = [np.array(msg.z[:4]), -np.array(msg.x[:4]), np.array(msg.y[:4])]
        self.foot_endpoints[0, :] += [0.09067, 0.09067, -0.09067, -0.09067]
        self.foot_endpoints[1, :] += [-0.085, 0.085, -0.085, 0.085]
        self.foot_endpoints[2, :] += [0.0, 0.0, 0.01, 0.01]
        if msg.z[4] > 180:
            msg.z[4] -= 360
        if msg.z[5] > 180:
            msg.z[5] -= 360
        self.lumbar_angles = np.array([msg.z[4], -(msg.y[4]-90)])
        self.head_angles = np.array([-msg.z[5]+30])

class BodyPoseSubscriber(Node):
    def __init__(self):
        super().__init__('body_pose_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/body_pose',
            self.listener_callback,
            10
        )
        self.mouth_angles = np.zeros(1)
        self.get_logger().info('BodyPose Subscriber has been started')

    def listener_callback(self, msg: String):
        if msg.data == 'open':
            self.mouth_angles[0] += 0.5
        elif msg.data == 'close':
            self.mouth_angles[0] -= 0.5
        # print(self.mouth_angles)
        

class SwitchModeService(Node):

    def __init__(self):
        super().__init__('switch_mode_service')
        self.mode = 'idle'
        self.srv = self.create_service(UnityAnimateService, 'UnityAnimate_srv', self.switch_mode_callback)

    def switch_mode_callback(self, request, response):
        print('received request', request.mode, type(request.mode))
        self.mode = request.mode
        response.response = "success"
        return response

class RobotState:
    def __init__(self):
        self.config = RobotConfiguration()
        self.ticks = 0                    
        self.height = -0.10
        self.foot_locations = self.config.default_stance
        self.joint_angles = np.zeros((3, 4))
        self.last_goal = np.zeros(4) 

class Command:
    def __init__(self):
        self.horizontal_velocity = np.array([0.0, 0.0])
        self.yaw_rate = 0
        self.height = -0.10

class RobotControl:
    def __init__(self):
        self.cmd_vel = CmdVelSubscriber()
        self.joint_states = JointStatesSubscriber()
        self.body_pose = BodyPoseSubscriber()
        self.JointStatesPublisher = JointStatesPublisher()
        self.switch_mode_service = SwitchModeService()
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.cmd_vel)
        self.executor.add_node(self.joint_states)
        self.executor.add_node(self.JointStatesPublisher)
        self.executor.add_node(self.switch_mode_service)
        self.executor.add_node(self.body_pose)

        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        self.config = RobotConfiguration()
        self.gait_controller = GaitController(self.config)
        self.swing_controller = SwingController(self.config)
        self.stance_controller = StanceController(self.config)
        self.inverse_kinematics = InverseKinematics(self.config)
        self.forward_kinematics = ForwardKinematics(self.config)

        self.state = RobotState()
        self.command = Command()
        self.control_cmd = ControlCmd()

        self.mode_threads = {
            'puppy_move': None,
            'play': None,
            'connect': None
        }
        self.stop_events = {
            'puppy_move': threading.Event(),
            'play': threading.Event(),
            'connect': threading.Event()
        }
        self.current_mode = 'idle'

    def get_vel_data(self):
        self.command.horizontal_velocity = np.array([self.cmd_vel.linear_x, self.cmd_vel.linear_y])
        self.command.yaw_rate = self.cmd_vel.angular_z

    def __del__(self):
        pass

    def cleanup(self):
        self.executor.shutdown()

    def step_gait(self, state, command):
        contact_modes = self.gait_controller.contacts(state.ticks)
        new_foot_locations = np.zeros((3, 4))

        for leg_index in range(4):
            contact_mode = contact_modes[leg_index]
            # foot_location = state.foot_locations[:, leg_index]
            if contact_mode == 1:  
                new_location = self.stance_controller.next_foot_location(leg_index, state, command)
            else: 
                swing_proportion = (
                    self.gait_controller.subphase_ticks(state.ticks) / self.config.swing_ticks
                )
                new_location = self.swing_controller.next_foot_location(
                    swing_proportion, leg_index, state, command
                )
            new_foot_locations[:, leg_index] = new_location

        return new_foot_locations, contact_modes

    def mode_handler(self):
        """Main mode handler that manages thread creation and termination"""
        while True:
            new_mode = self.switch_mode_service.mode
            # If mode has changed
            if new_mode != self.current_mode:
                print(f"Switching from {self.current_mode} to {new_mode}")
                
                # Stop all current threads
                for mode, event in self.stop_events.items():
                    event.set()
                
                # Wait for threads to terminate
                for mode, thread in self.mode_threads.items():
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=1.0)  # Wait up to 1 second for clean termination
                
                # Reset all stop events
                for mode in self.stop_events:
                    self.stop_events[mode].clear()
                
                # Start new thread based on mode
                if new_mode == 'puppy_move':
                    self.control_cmd.enable_all_motor()
                    self.mode_threads['puppy_move'] = threading.Thread(
                        target=self.puppy_move_thread, 
                        args=(self.stop_events['puppy_move'],)
                    )
                    self.mode_threads['puppy_move'].start()
                    
                elif new_mode == 'play':
                    self.control_cmd.enable_all_motor()
                    self.mode_threads['play'] = threading.Thread(
                        target=self.puppy_move_unity_thread, 
                        args=(self.stop_events['play'],)
                    )
                    self.mode_threads['play'].start()
                    
                elif new_mode == 'connect':
                    self.control_cmd.disable_all_motor()
                    self.mode_threads['connect'] = threading.Thread(
                        target=self.unity_mirror_thread, 
                        args=(self.stop_events['connect'],)
                    )
                    self.mode_threads['connect'].start()
                    
                elif new_mode == 'idle':
                    self.control_cmd.disable_all_motor()
                    # No thread to start for idle mode
                
                self.current_mode = new_mode
            
            time.sleep(0.1)  # Check for mode changes every 100ms
    
    def puppy_move_thread(self, stop_event):
        """Thread function for puppy_move mode"""
        print("Starting puppy_move thread")
        while not stop_event.is_set():
            self.get_vel_data()

            # Asynchronously calculate foot placements
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_foot_locations = executor.submit(self.step_gait, self.state, self.command)

            self.state.joint_angles = self.inverse_kinematics.four_legs_inverse_kinematics(
                self.state.foot_locations
            )

            # Wait for foot placements to be ready
            self.state.foot_locations, self.contact_modes = future_foot_locations.result()
            goal = (
                self.state.joint_angles * 180 / 3.14 * DEGREE_TO_SERVO
                + self.config.leg_center_position
            )

            lumbar_pos = np.zeros(2)
            head_pos = np.zeros(1)
            mouth_pos = self.body_pose.mouth_angles * DEGREE_TO_SERVO

            # Only update motor positions if the goal has significantly changed
            if np.linalg.norm(goal - self.state.last_goal) > self.config.goal_change_threshold:
                self.control_cmd.motor_position_control(goal, lumbar_pos, head_pos, mouth_pos)
                self.state.last_goal = goal
            self.state.ticks += 1

            # Add a small sleep to prevent CPU overuse
            time.sleep(0.01)
        
        print("Exiting puppy_move thread")
    
    def puppy_move_unity_thread(self, stop_event):
        """Thread function for puppy_move_unity mode"""
        print("Starting puppy_move_unity thread")
        while not stop_event.is_set():
            # print('playing')
            self.get_vel_data()

            # Asynchronously calculate foot placements
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_foot_locations = executor.submit(self.step_gait, self.state, self.command)

            self.state.joint_angles = self.inverse_kinematics.four_legs_inverse_kinematics(
                self.joint_states.foot_endpoints
            )

            # Wait for foot placements to be ready
            self.state.foot_locations, self.contact_modes = future_foot_locations.result()
            goal = (
                self.state.joint_angles * 180 / 3.14 * DEGREE_TO_SERVO
                + self.config.leg_center_position
            )
            print(self.joint_states.head_angles)
            lumbar_pos = self.joint_states.lumbar_angles * DEGREE_TO_SERVO
            head_pos = self.joint_states.head_angles * DEGREE_TO_SERVO
            mouth_pos = self.body_pose.mouth_angles * DEGREE_TO_SERVO
            
            # Only update motor positions if the goal has significantly changed
            if np.linalg.norm(goal - self.state.last_goal) > self.config.goal_change_threshold:
                self.control_cmd.motor_position_control(goal, lumbar_pos, head_pos, mouth_pos)
                self.state.last_goal = goal
            self.state.ticks += 1

            # Add a small sleep to prevent CPU overuse
            # time.sleep(0.01)
        
        print("Exiting puppy_move_unity thread")
    
    def unity_mirror_thread(self, stop_event):
        """Thread function for unity_mirror mode"""
        print("Starting unity_mirror thread")
        t0 = time.time()
        while not stop_event.is_set():
            position = np.zeros(16)

            self.control_cmd.dynamixel.updateMotorData()
            # print('motor_update')

            for i, motor in enumerate(self.control_cmd.dynamixel.motors):
                position[i] = motor.PRESENT_POSITION_value
            
            motor_angles = position/DEGREE_TO_SERVO-180
            self.leg_angles = np.zeros([3, 4])
            
            for i in range(4):
                self.leg_angles[:, i] = self.forward_kinematics.leg_explicit_forward_kinematics_2(
                    motor_angles[3*i], motor_angles[3*i+1], motor_angles[3*i+2], i
                )
            
            joint_state_msg = JointState()
            joint_state_msg.name = [
                'FR', 'FL', 'RR', 'RL', 'LUMBAR', 'HEAD', 'MOUTH'
            ]
            joint_state_msg.x = self.leg_angles[0, :].tolist()  # First row (all columns)
            joint_state_msg.y = self.leg_angles[1, :].tolist()  # Second row (all columns)
            joint_state_msg.z = self.leg_angles[2, :].tolist()  # Third row (all columns)
            joint_state_msg.x.append(0.0)  # Lumbar X
            joint_state_msg.y.append(motor_angles[12])  # Lumbar Y
            joint_state_msg.z.append(motor_angles[13])  # Lumbar Z
            joint_state_msg.x.append(0.0)  # Head X
            joint_state_msg.y.append(motor_angles[14])  # Head Y
            joint_state_msg.z.append(0.0)  # Head Z
            joint_state_msg.x.append(0.0)  # Mouth X
            joint_state_msg.y.append(motor_angles[15])  # Mouth Y
            joint_state_msg.z.append(0.0)  # Mouth Z
            self.JointStatesPublisher.publisher_.publish(joint_state_msg)

            # Add a small sleep to prevent CPU overuse
            # time.sleep(0.01)
            t1 = time.time()
            # print(1/(t1-t0))
            t0 = t1
        
        print("Exiting unity_mirror thread")

class ControlCmd:
    """Low-level control of Dynamixel motors."""
    def __init__(self):
        self.setup_dynamixel()
        self.setup_motors()
        self.initialize_motor_states()
        self.config = RobotConfiguration()
        self.forward_kinematics = ForwardKinematics(self.config)
        
        self.walking_freq = 2000  # 2000Hz

    def setup_dynamixel(self):
        self.robot_dynamixel = RobotDynamixel()
        self.dynamixel = DXL_Communication(self.robot_dynamixel.DEVICE_NAME, self.robot_dynamixel.B_RATE)
        self.dynamixel.activateDXLConnection()
        
    def setup_motors(self):
        motor_names = ['FR_higher', 'FR_lower', 'FR_hip',  
                        'FL_higher', 'FL_lower', 'FL_hip',
                       'RR_higher', 'RR_lower', 'RR_hip',
                       'RL_higher', 'RL_lower', 'RL_hip',
                       'lumbar_y', 'lumbar_z',
                       'head', 'mouth']

        self.motor_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.motors = {name: self.dynamixel.createMotor(name, motor_number=id_) 
                    for name, id_ in zip(motor_names, self.motor_ids)}

        self.leg_motor_list = [
            [self.motors['FR_hip'], self.motors['FL_hip'], self.motors['RR_hip'], self.motors['RL_hip']],
            [self.motors['FR_higher'], self.motors['FL_higher'], self.motors['RR_higher'], self.motors['RL_higher']],
            [self.motors['FR_lower'], self.motors['FL_lower'], self.motors['RR_lower'], self.motors['RL_lower']]
        ]
    
    def initialize_motor_states(self):
        self.dynamixel.addAllBuckPrarmeter()
        print('add all params')
        # self.dynamixel.rebootAllMotor()
        self.dynamixel.updateMotorData()
        # self.enable_all_motor()
        self.joint_position = np.zeros(12)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        self.disable_all_motor()
        self.dynamixel.closeHandler()

    def enable_all_motor(self):
        for motor in self.motors.values():
            motor.enableMotor()

    def disable_all_motor(self):
        for motor in self.motors.values():
            motor.disableMotor()

    def read_all_motor_data(self):
        self.update_joint_state()
        return print(self.joint_position)
    
    def update_joint_state(self):
        self.dynamixel.updateMotorData()
        self.joint_position = np.zeros((3, 4))
        self.joint_position[0] = [motor.PRESENT_POSITION_value for motor in self.leg_motor_list[0]]
        self.joint_position[1] = [motor.PRESENT_POSITION_value for motor in self.leg_motor_list[1]]
        self.joint_position[2] = [motor.PRESENT_POSITION_value for motor in self.leg_motor_list[2]]

    def reset_to_original(self, position=None):
        self.motor_position_control()

    def motor_position_control(self, position=None, lumbar_pos=None, head_pos=None, mouth_pos=None):
        print(head_pos)
        if position is None:
            position = [[2048 ,2048, 2048, 2048],
                        [1992, 2047, 2092, 2099],
                        [2048 ,2048, 2048, 2048]]
        
        for i, motor_list in enumerate(self.leg_motor_list):
            for j, motor in enumerate(motor_list):
                motor.writePosition(int(position[i][j]))
        
        # lock lumbar
        self.motors['lumbar_y'].writePosition(int(2048+lumbar_pos[0]))
        self.motors['lumbar_z'].writePosition(int(2048+lumbar_pos[1]))
        self.motors['head'].writePosition(int(2048+head_pos[0]))
        self.motors['mouth'].writePosition(int(2048+mouth_pos[0]))
        self.dynamixel.sentAllCmd()

def main():
    rclpy.init()
    robot_control = RobotControl()

    command_dict = {
        # "s":robot_control.puppy_move,
        # "u":robot_control.puppy_move_unitiy,
        # "r":robot_control.unity_mirror,
        "a":robot_control.mode_handler
    }

    atexit.register(robot_control.cleanup)

    while True:
        try:
            cmd = input("CMD : ")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                break
        except Exception as e:
            robot_control.cleanup()
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()