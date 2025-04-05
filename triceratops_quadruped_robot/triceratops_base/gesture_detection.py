#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import argparse
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys, os
import threading
from flask import Flask, Response
import socket
# Import the Unity service from the same path as the first code
sys.path.append(os.path.expanduser("~/UnityRos2_ws/src/unity_robotics_demo_msgs"))
from unity_robotics_demo_msgs.srv import UnityAnimateService

# Initialize Flask app for streaming
app = Flask(__name__)

class GestureControlNode(Node):
    def __init__(self, camera_url, stream_enabled=False, stream_port=5000):
        super().__init__('gesture_control_node')
        
        # Initialize ROS2 publisher for debugging
        self.gesture_pub = self.create_publisher(String, 'detected_gesture', 10)
        
        # Initialize service client similar to the first code
        self.cli = self.create_client(UnityAnimateService, 'UnityAnimate_srv')
        self.req = UnityAnimateService.Request()
        
        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize video capture
        self.get_logger().info(f"Connecting to stream at {camera_url}")
        self.cap = cv2.VideoCapture(camera_url)
        
        if not self.cap.isOpened():
            self.get_logger().info("Failed to open the stream. Trying RTSP...")
            rtsp_url = f"rtsp://{camera_url.split('//')[1]}"
            self.cap = cv2.VideoCapture(rtsp_url)
            
        if not self.cap.isOpened():
            self.get_logger().info("Failed to open the stream. Trying with mjpg...")
            mjpg_url = f"{camera_url}/mjpg/video.mjpg"
            self.cap = cv2.VideoCapture(mjpg_url)
        
        if not self.cap.isOpened():
            self.get_logger().error("Failed to connect to the stream. Please check the URL.")
            return
        
        # Set buffer size to minimum to avoid lag
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.get_logger().info("Connected to the stream successfully!")
        
        # Frame buffer for streaming
        self.frame_lock = threading.Lock()
        self.frame_buffer = None
        self.processed_frame_buffer = None
        
        # Gesture tracking variables
        self.current_gesture = "No gesture detected"
        self.prev_gesture = "No gesture detected"
        self.current_mode = ""
        
        # Create a timer for processing frames
        self.create_timer(0.05, self.process_frame)  # 20 FPS
        
        # Start streaming server if enabled
        self.stream_enabled = stream_enabled
        if stream_enabled:
            # Get local IP address
            self.local_ip = self.get_local_ip()
            self.stream_port = stream_port
            # Start Flask server in a separate thread
            threading.Thread(target=self.start_streaming_server, daemon=True).start()
            self.get_logger().info(f"Streaming server started at http://{self.local_ip}:{stream_port}")
        
        self.get_logger().info("Gesture Control Node Initialized")
    
    def get_local_ip(self):
        """Get the local IP address of this machine"""
        try:
            # Create a socket connection to determine the local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Doesn't need to be reachable
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return '127.0.0.1'  # Fallback to localhost
            
    def start_streaming_server(self):
        """Start the Flask streaming server"""
        app.node = self  # Store reference to node instance
        app.run(host='0.0.0.0', port=self.stream_port, threaded=True)
        
    def generate_frames(self):
        """Generator function for video streaming"""
        while True:
            # Get the latest processed frame with annotations
            with self.frame_lock:
                if self.processed_frame_buffer is None:
                    continue
                frame = self.processed_frame_buffer.copy()
            
            # Encode the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    def send_service_request(self, mode):
        if not self.cli.service_is_ready():
            self.get_logger().info('Service not ready')
            return False
            
        self.req.mode = mode
        self.future = self.cli.call_async(self.req)
        self.get_logger().info(f"Service request sent: {mode}")
        return True
    
    def process_frame(self):
        try:
            # Flush the buffer by reading multiple frames
            for _ in range(5):  # Read a few frames to clear the buffer
                ret, _ = self.cap.read()
                if not ret:
                    break
                    
            # Now read the most recent frame
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("Failed to get frame. Stream may have ended.")
                return
            
            # Store the original frame in buffer
            with self.frame_lock:
                self.frame_buffer = frame.copy()
            
            # Create a copy for gesture detection display
            gesture_frame = frame.copy()
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame for hand detection
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks and detect gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks on the gesture frame for streaming
                    self.mp_drawing.draw_landmarks(
                        gesture_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Detect gesture based on landmarks
                    self.current_gesture = self.detect_gesture(hand_landmarks)
            else:
                self.current_gesture = "No gesture detected"
            
            # Add text overlay for streaming
            if self.stream_enabled:
                cv2.putText(gesture_frame, f"Gesture: {self.current_gesture}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(gesture_frame, f"Mode: {self.current_mode}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Store the processed frame in buffer for streaming
                with self.frame_lock:
                    self.processed_frame_buffer = gesture_frame.copy()
            
            # Publish detected gesture
            msg = String()
            msg.data = self.current_gesture
            self.gesture_pub.publish(msg)
            
            # Check if gesture changed and execute corresponding action
            if self.current_gesture != self.prev_gesture:
                self.get_logger().info(f"Gesture changed: {self.current_gesture}")
                
                # Handle Palm gesture - send "play" command like button 'Y' in the first code
                if self.current_gesture == "Palm" and self.current_mode != "puppy_move":
                    self.current_mode = "puppy_move"
                    self.send_service_request("puppy_move")
                    self.get_logger().info("Palm detected: Sending 'puppy_move' command")
                # Just log other gestures to console without sending service requests
                elif self.current_gesture == "Fist" and self.current_mode != "connect":
                    self.current_mode = "connect"
                    self.send_service_request("connect")
                    self.get_logger().info("Palm detected: Sending 'connect' command")
                elif self.current_gesture == "Victory/Peace Sign":
                    self.get_logger().info("Victory/Peace Sign detected: No action taken")
                elif self.current_gesture == "Thumbs Up":
                    self.get_logger().info("Thumbs Up detected: No action taken")
                elif self.current_gesture == "Pointing" and self.current_mode != "play":
                    self.current_mode = "play"
                    self.send_service_request("play")
                    self.get_logger().info("Palm detected: Sending 'play' command")
                    
                self.prev_gesture = self.current_gesture
                
        except Exception as e:
            self.get_logger().error(f"An error occurred: {e}")

    def detect_gesture(self, hand_landmarks):
        """
        Detect gestures based on hand landmarks
        """
        # Get landmark positions
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        
        # Thumb tip is landmark 4
        thumb_tip = landmarks[4]
        # Index finger tip is landmark 8
        index_tip = landmarks[8]
        # Middle finger tip is landmark 12
        middle_tip = landmarks[12]
        # Ring finger tip is landmark 16
        ring_tip = landmarks[16]
        # Pinky tip is landmark 20
        pinky_tip = landmarks[20]
        
        # Palm center (approximately)
        wrist = landmarks[0]
        
        # Calculate distances between fingertips and palm
        thumb_dist = self.distance(thumb_tip, wrist)
        index_dist = self.distance(index_tip, wrist)
        middle_dist = self.distance(middle_tip, wrist)
        ring_dist = self.distance(ring_tip, wrist)
        pinky_dist = self.distance(pinky_tip, wrist)
        
        # Calculate vertical positions relative to palm
        thumb_y_pos = thumb_tip[1] - wrist[1]
        index_y_pos = index_tip[1] - wrist[1]
        middle_y_pos = middle_tip[1] - wrist[1]
        ring_y_pos = ring_tip[1] - wrist[1]
        pinky_y_pos = pinky_tip[1] - wrist[1]
        
        # Check for different gestures
        thred = 0.4
        
        # Fist - all fingertips close to palm
        if (index_dist < thred and middle_dist < thred and 
            ring_dist < thred and pinky_dist < thred):
            return "Fist"
        
        # Thumbs up - thumb extended upward, other fingers closed
        if (thumb_y_pos < -0.1 and index_dist < thred and 
            middle_dist < thred and ring_dist < thred and pinky_dist < thred):
            return "Thumbs Up"
        
        # Victory/Peace sign - index and middle fingers extended, others closed
        if (index_dist > thred and middle_dist > thred and 
            ring_dist < thred and pinky_dist < thred):
            return "Victory/Peace Sign"
        
        # Pointing - only index finger extended
        if (index_dist > thred and middle_dist < thred and 
            ring_dist < thred and pinky_dist < thred):
            return "Pointing"
        
        # Open palm - all fingers extended
        if (index_dist > thred and middle_dist > thred and 
            ring_dist > thred and pinky_dist > thred):
            return "Palm"
        
        # Default if no specific gesture is detected
        return "Unknown gesture"

    def distance(self, p1, p2):
        """Calculate 3D distance between two points"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5
    
    def __del__(self):
        # Release resources
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()

# Flask routes
@app.route('/')
def index():
    """Video streaming home page"""
    return """
    <html>
    <head>
        <title>Gesture Detection Stream</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
            h1 { color: #333; }
            .video-container { margin-top: 20px; }
            img { max-width: 100%; border: 2px solid #333; }
        </style>
    </head>
    <body>
        <h1>Gesture Detection Stream</h1>
        <div class="video-container">
            <img src="/video_feed" />
        </div>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(app.node.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Gesture detection with ROS2 integration')
    parser.add_argument('--url', type=str, default='http://192.168.41.106:81/stream',
                        help='URL of the IP camera stream (default: http://192.168.41.106:81/stream)')
    parser.add_argument('--stream', action='store_true',
                        help='Enable video streaming server')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for the streaming server (default: 5000)')
    parsed_args = parser.parse_args()
    
    rclpy.init(args=args)
    
    # Create and run the node
    node = GestureControlNode(parsed_args.url, parsed_args.stream, parsed_args.port)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup on shutdown
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()