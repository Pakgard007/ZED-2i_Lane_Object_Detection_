import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32

class DriveControlNode(Node):
    def __init__(self):
        super().__init__('drive_control_node')

        # Subscriptions: รับความเร็วและมุมพวงมาลัยแยกกัน
        self.create_subscription(Float32, '/cmd_speed', self.cmd_speed_callback, 10)
        self.create_subscription(Float32, '/cmd_angle', self.cmd_angle_callback, 10)
        self.create_subscription(String, '/drive_status', self.drive_status_callback, 10)

        # Publisher
        self.final_cmd_pub = self.create_publisher(Twist, '/final_cmd_vel', 10)

        # Internal state
        self.latest_cmd_vel = Twist()  # สร้าง Twist จาก speed + angle
        self.cmd_speed = 0.0
        self.cmd_angle = 0.0

        self.current_status = "go"
        self.current_speed = 1.8  # start at max
        self.max_speed = 2.0
        self.deceleration_step = 0.1  # reduce 0.1 m/s per callback

        self.timer = self.create_timer(0.1, self.update_final_cmd)  # 10 Hz

        self.get_logger().info("Drive Control Node started.")

    def cmd_speed_callback(self, msg):
        self.cmd_speed = msg.data
        self.update_cmd_vel()

    def cmd_angle_callback(self, msg):
        self.cmd_angle = msg.data
        self.update_cmd_vel()

    def update_cmd_vel(self):
        # รวมค่าจาก /cmd_speed และ /cmd_angle มาเก็บเป็น Twist
        self.latest_cmd_vel.linear.x = self.cmd_speed
        self.latest_cmd_vel.angular.z = self.cmd_angle

    def drive_status_callback(self, msg):
        self.current_status = msg.data.lower().strip()

    def update_final_cmd(self):
        cmd = Twist()

        if self.current_status == "go":
            self.current_speed = self.max_speed

        elif self.current_status == "slow":
            self.current_speed = max(0.0, self.current_speed - self.deceleration_step)

        elif self.current_status == "brake":
            self.current_speed = 0.0

        else:
            self.get_logger().warn(f"Unknown drive_status: {self.current_status}. Using brake.")
            self.current_speed = 0.0

        # Apply current speed and angle
        cmd.linear.x = min(self.current_speed, abs(self.latest_cmd_vel.linear.x)) * np.sign(self.latest_cmd_vel.linear.x)
        cmd.angular.z = self.latest_cmd_vel.angular.z if cmd.linear.x != 0.0 else 0.0

        self.final_cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = DriveControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
