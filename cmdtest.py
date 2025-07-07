import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import math
import time

class TestSpeedAnglePublisher(Node):
    def __init__(self):
        super().__init__('test_speed_angle_publisher')

        self.speed_pub = self.create_publisher(Float32, '/cmd_speed', 10)
        self.angle_pub = self.create_publisher(Float32, '/cmd_angle', 10)

        self.timer = self.create_timer(0.5, self.publish_test_values)  # ส่งทุก 0.5 วินาที
        self.counter = 0.0

        self.get_logger().info("✅ TestSpeedAnglePublisher started.")

    def publish_test_values(self):
        # ทดสอบ: ความเร็ว 1.8 m/s คงที่, มุมแกว่งไปมาระหว่าง -0.5 ถึง +0.5 rad
        speed_msg = Float32()
        angle_msg = Float32()

        speed_msg.data = 1.8

        angle_msg.data = 0.5 * math.sin(self.counter)  # มุมแกว่งขึ้นลงเหมือนรถเลี้ยว
        self.counter += 0.2

        self.speed_pub.publish(speed_msg)
        self.angle_pub.publish(angle_msg)

        self.get_logger().info(f"📤 Published speed: {speed_msg.data:.2f} | angle: {angle_msg.data:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = TestSpeedAnglePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
