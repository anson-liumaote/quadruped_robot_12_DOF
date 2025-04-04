import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time
# from champ_interfaces.srv import SetMode
import sys, os
sys.path.append(os.path.expanduser("~/UnityRos2_ws/src/unity_robotics_demo_msgs"))
from unity_robotics_demo_msgs.msg import JointState
from unity_robotics_demo_msgs.srv import UnityAnimateService


class TriceratopsControlClient(Node):
    def __init__(self):
        super().__init__('JoyControlClient')
        self.create_subscription(Joy, "joy", self.joy_callback, 1)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self.body_pub = self.create_publisher(String, 'body_pose', 1)
        self.cli = self.create_client(UnityAnimateService, 'UnityAnimate_srv')
        self.req = UnityAnimateService.Request()
        
        self.current_mode = ""

        self.linear_x_scale = 0.1
        self.linear_y_scale = 0.08
        self.angular_scale = 1
        feq = 500
        self.joy = None
        self.cmd = None
        
        self.timer = self.create_timer(1/feq, self.timer_callback)
        self.get_logger().info("Client Initialized")
        
    def joy_callback(self, data: Joy):
        
        if(data.buttons[4]==1 and data.buttons[5]==1 and data.buttons[2]==1):
            self.current_mode = "stop"
            time.sleep(0.01)
        elif(data.buttons[3]==1 and self.current_mode != "y"):
            self.current_mode = "y"
            self.req.mode = "play"
            self.future = self.cli.call_async(self.req)
            time.sleep(0.01)
        elif(data.buttons[0]==1 and self.current_mode != "a"):
            self.current_mode = "a"
            self.req.mode = "connect"
            self.future = self.cli.call_async(self.req)
            time.sleep(0.01)            
        elif(data.buttons[2]==1):
            self.current_mode = "x"
            time.sleep(0.01)
        elif(data.buttons[1]==1 and self.current_mode != "b"):
            self.current_mode = "b"
            self.req.mode = "puppy_move"
            self.future = self.cli.call_async(self.req)
            time.sleep(0.01)
        elif(data.buttons[6]==1 and self.current_mode != "start"):
            self.current_mode = "open"
            time.sleep(0.01)
        elif(data.buttons[7]==1 and self.current_mode != "start"):
            self.current_mode = "close"
            time.sleep(0.01)
        elif(data.axes[3]==1 and self.current_mode != "up"):
            self.current_mode = "up"
            time.sleep(0.01)
        elif(data.axes[3]==-1 and self.current_mode != "down"):
            self.current_mode = "down"
            time.sleep(0.01)
        elif(data.axes[2]==1 and self.current_mode != "left"):
            self.current_mode = "left"
            time.sleep(0.1)
        elif(data.axes[2]==-1 and self.current_mode != "right"):
            self.current_mode = "right"
            time.sleep(0.01)
        else:
            self.current_mode = " "
        self.get_logger().info(f"The current mode {self.current_mode}")
        self.joy = data

    def timer_callback(self):
        vel = Twist()
        pose = String()
        pose.data = str(self.current_mode)
        
        if self.joy is not None:
            if self.joy.axes[1]>0.001:
                vel.linear.x = self.joy.axes[1]*self.linear_x_scale
            elif self.joy.axes[1]<-0.001:
                vel.linear.x = self.joy.axes[1]*self.linear_x_scale
            if self.joy.axes[0]>0.001:
                vel.linear.y = self.joy.axes[0]*self.linear_y_scale
            elif self.joy.axes[0]<-0.001:
                vel.linear.y = self.joy.axes[0]*self.linear_y_scale
            if self.joy.axes[2]>0.001:
                vel.angular.z = self.joy.axes[2]*self.angular_scale
            elif self.joy.axes[2]<-0.001:
                vel.angular.z = self.joy.axes[2]*self.angular_scale

        self.cmd_pub.publish(vel)
        self.body_pub.publish(pose)
        
def main(args=None):
    rclpy.init(args=args)

    controller = TriceratopsControlClient()

    rclpy.spin(controller)

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
