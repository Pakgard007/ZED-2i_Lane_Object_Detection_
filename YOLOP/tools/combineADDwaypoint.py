import math
import time
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- โหลด Waypoints แบบ Absolute -----------------------
df = pd.read_csv("/home/mag/waypoint/GPS_with_PositionXY_updated.csv")
waypoints = df.to_dict(orient="records")

# ----------------------- ฟังก์ชันช่วย -----------------------
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def compute_control(current_x, current_y, current_theta, goal_x, goal_y, goal_theta):
    dx = goal_x - current_x
    dy = goal_y - current_y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    target_theta = math.atan2(dy, dx)
    heading_error = normalize_angle(target_theta - current_theta)

    # Steering
    if abs(heading_error) < 0.1:
        steering = 0  # ตรง
    elif heading_error > 0:
        steering = 1  # ขวา
    else:
        steering = -1 # ซ้าย

    # Speed
    if distance > 2.0:
        speed = 0.5
    elif distance > 0.5:
        speed = 0.3
    else:
        speed = 0.0

    return speed, steering, distance

# ----------------------- Simulation -----------------------
car_x = waypoints[0]["x_back"]
car_y = waypoints[0]["y_back"]
car_theta = waypoints[0]["orientation"]

trajectory_x = [car_x]
trajectory_y = [car_y]
trajectory_theta = [car_theta]
speed_list = [0.0]

current_wp_index = 0

print("\n🚗 เริ่มจำลองการเคลื่อนที่ตาม Waypoint (ใช้ World Coordinate)\n")

while current_wp_index < len(waypoints):
    wp = waypoints[current_wp_index]
    speed, steering, dist = compute_control(car_x, car_y, car_theta,
                                            wp["x_back"], wp["y_back"], wp["orientation"])

    print(f"🔸 ไปยัง {wp['name']} | ระยะ: {dist:.2f} m | speed: {speed:.2f} | steering: {steering}")

    if dist < 0.5:
        print(f"✅ ถึง {wp['name']} แล้ว\n")
        current_wp_index += 1
        continue

    # จำลองการเคลื่อนที่ (update position)
    dt = 0.1  # ความถี่ 10 Hz
    car_theta += steering * 0.1 * dt
    car_theta = normalize_angle(car_theta)
    car_x += speed * math.cos(car_theta) * dt
    car_y += speed * math.sin(car_theta) * dt

    trajectory_x.append(car_x)
    trajectory_y.append(car_y)
    trajectory_theta.append(car_theta)
    speed_list.append(speed)

    time.sleep(dt)

# ----------------------- แสดงผล -----------------------
print("🟢 สิ้นสุดการเดินทาง\n")
plt.figure(figsize=(10, 12))

# 1. Heatmap จาก speed
sc = plt.scatter(trajectory_x, trajectory_y, c=speed_list, cmap='coolwarm', label='Speed (m/s)', s=10)
plt.colorbar(sc, label='Speed (m/s)')

# 2. ทิศทางของรถ (Heading Arrow)
for i in range(0, len(trajectory_x), 20):
    plt.arrow(
        trajectory_x[i], trajectory_y[i],
        0.5 * math.cos(trajectory_theta[i]),
        0.5 * math.sin(trajectory_theta[i]),
        head_width=0.5, head_length=1, fc='green', ec='green'
    )

# 3. Waypoints
plt.plot([wp['x_back'] for wp in waypoints], [wp['y_back'] for wp in waypoints], 'rx--', label='Waypoints')
for wp in waypoints:
    plt.text(wp["x_back"], wp["y_back"], wp["name"], fontsize=6, color='red')

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Vehicle Path Following Absolute Waypoints + Heading + Speed")
plt.legend()
plt.grid(True)
plt.axis('equal')
print(f"📍 ตำแหน่งสุดท้ายของรถ:")
print(f"   x = {car_x:.2f} m")
print(f"   y = {car_y:.2f} m")
print(f"   theta (heading) = {math.degrees(car_theta):.2f}°")
plt.show()
