import cv2
import numpy as np
import time
import queue
import threading
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
from dronekit import connect, VehicleMode, LocationGlobalRelative

# ----------------------- SHAPE DETECTION AND IMAGE PROCESSING PART -----------------------

SHAPE_TIMEOUT = 0.5

#Queue for detected shapes
shape_queue = queue.Queue()
#Dictionary for detected shapes
detected_shapes = {}

class CameraSubscriber:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('camera_subscriber', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/webcam/image_raw', Image, self.image_callback)
        self.latest_frame = None  # Son alınan çerçeveyi saklamak için

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_frame = cv_image  # Son alınan çerçeveyi güncelle
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")


def add_detected_shape_queue(shape_type, shape_color, position):
    shape_info = {
        "type": shape_type,
        "color": shape_color,
        "position": position,
        "time_stamp": time.time()
    }
    shape_queue.put(shape_info)

def remove_old_shapes():
    while True:
        time.sleep(1)
        current_time = time.time()
        to_remove = [key for key, shape in detected_shapes.items() if current_time - shape["time_stamp"] > SHAPE_TIMEOUT]

        for key in to_remove:
            del detected_shapes[key]


def update_add_detected_shape(shape_type, shape_color, shape_position):
    shape_id = f"{shape_type}_{shape_color}"
    shape_info = {
        "id": shape_id,
        "type": shape_type,
        "color": shape_color,
        "position": shape_position,
        "time_stamp": time.time()
    }
    detected_shapes[shape_id] = shape_info

def process_detected_shapes():
    while True:
        shape_info = shape_queue.get()
        update_add_detected_shape(
            shape_type=shape_info["type"],
            shape_color=shape_info["color"],
            shape_position=shape_info['position']
        )

def getDominantColor(x, y, w, h, clean_frame):
    shape_colors = ('b', 'g', 'r')
    shape_dominant_index = None
    histograms = {}
    shape_dominant_color = {}
    color_names = {
        'r': "Red",
        'g': "Green",
        'b': "Blue",
    }

    crop_image = clean_frame[y:y+h, x:x+w]

    for i, color in enumerate(shape_colors):
        color_hist = cv2.calcHist([crop_image], [i], None, [256], [0, 256])
        histograms[color] = color_hist

    for color in histograms:
        max_intensity = np.argmax(histograms[color])
        shape_dominant_color[color] = max_intensity

    shape_dominant_index = max(shape_dominant_color, key=shape_dominant_color.get)
    return color_names[shape_dominant_index]

def getBorderDominantColor(x, y, w, h, approx, clean_frame):
    border_dominant_color = {}
    histograms = {}
    border_colors = ('b', 'g', 'r')
    border_dominant_index = None
    color_names = {
        'r': "Red",
        'g': "Green",
        'b': "Blue",
    }

    crop_image_outline = clean_frame[y:h + y, x:w + x]

    border_width = 100  
    outline = clean_frame[y:y + h, x:x + w] 

    mask = np.zeros_like(outline, dtype=np.uint8)
    cv2.drawContours(mask, [approx - [x, y]], -1, (255, 255, 255), thickness=border_width)  
    edge_only = cv2.bitwise_and(outline, outline, mask=mask[:, :, 0])

    cv2.imshow("Edge Only", edge_only)

    for i, color in enumerate(border_colors):
        edge_color_hist = cv2.calcHist([crop_image_outline], [i], None, [256], [0, 256])
        histograms[color] = edge_color_hist

    for color in histograms:
        max_intensity = np.argmax(histograms[color])
        border_dominant_color[color] = max_intensity

    border_dominant_index = max(border_dominant_color, key=border_dominant_color.get)
    return [color_names[border_dominant_index]]

def get_full_screen_dominant_color(frame):
    _frame = frame.copy()
    channels = ('b', 'g', 'r')
    color_names = {'r': "Red", 'g': "Green", 'b': "Blue"}
    color_ratios = {}

    total_pixels = _frame.shape[0] * _frame.shape[1]
    b_channel, g_channel, r_channel = cv2.split(_frame)
    
    b_intensity = np.sum(b_channel) / (total_pixels * 255) * 100  
    g_intensity = np.sum(g_channel) / (total_pixels * 255) * 100
    r_intensity = np.sum(r_channel) / (total_pixels * 255) * 100

    general_intensity = (b_intensity + g_intensity + r_intensity) / 3

    color_ratios['b'] = b_intensity
    color_ratios['g'] = g_intensity
    color_ratios['r'] = r_intensity

    print(f"🔵 Blue: {color_ratios['b']:.2f}% | 🟢 Green: {color_ratios['g']:.2f}% | 🔴 Red: {color_ratios['r']:.2f}%")

    dominant_colors = [color for color in color_ratios if color_ratios[color] > 30]
    
    if dominant_colors:
        dominant_color = dominant_colors[0]
        dominant_color_intensity = color_ratios[dominant_color]
        if abs(dominant_color_intensity - general_intensity) >= 10:
            return color_names[dominant_colors[0]]

    return "No dominant color"  

def initalizeKeyBinds():  
    keybind = cv2.waitKey(1) & 0xFF 
    if keybind == ord('q'):
        cv2.destroyAllWindows()
        return True
    return False

def process_frame(frame):
    if frame is None:
        return frame
        
    clean_frame = frame.copy()
    clean2_frame = frame.copy()

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mavi renk aralığı
    blue_lower = np.array([95, 100, 50])  
    blue_upper = np.array([140, 255, 255])  

    # Kırmızı renk aralığı
    red_lower_1 = np.array([-2, 120, 50])     
    red_upper_1 = np.array([12, 255, 255])   
    red_lower_2 = np.array([165, 120, 50])   
    red_upper_2 = np.array([185, 255, 255])  

    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

    red_mask_1 = cv2.inRange(hsv_image, red_lower_1, red_upper_1)
    red_mask_2 = cv2.inRange(hsv_image, red_lower_2, red_upper_2)
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    overlap_mask = cv2.bitwise_and(blue_mask, red_mask)

    final_red = cv2.subtract(red_mask, overlap_mask)
    final_blue = cv2.subtract(blue_mask, overlap_mask)

    final_mask = cv2.bitwise_or(final_red, final_blue)
    cv2.imshow("Final Mask", final_mask)

    hsv_result = cv2.bitwise_and(frame, frame, mask=final_mask)
    
    gray_result = cv2.cvtColor(hsv_result, cv2.COLOR_BGR2GRAY)

    gray_result = cv2.medianBlur(gray_result, 7)
    gray_result = cv2.erode(gray_result, kernel=np.ones((5, 5), dtype=np.uint8), iterations=3)
    gray_result = cv2.morphologyEx(gray_result, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=3)

    gray_frame = cv2.cvtColor(clean2_frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    equalized_frame = cv2.equalizeHist(blurred_frame)
    _, otsu_thresh = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh_val, _ = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    w_edges = cv2.Canny(blurred_frame, otsu_thresh_val * 0.5, otsu_thresh_val)
    fields = cv2.Canny(gray_result, 100, 200)

    cv2.imshow("Second Frame", w_edges)

    full_screen_intense = get_full_screen_dominant_color(frame)
    print("Full Screen Intense: ", full_screen_intense)

    if full_screen_intense == "Blue":
        update_add_detected_shape("Hexagon", "Blue", (0, 0))
    elif full_screen_intense == "Red":
        update_add_detected_shape("Triangle", "Red", (0, 0))
        
    weight_contours, _ = cv2.findContours(w_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    field_contours, _ = cv2.findContours(fields, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled_frame = np.zeros_like(fields)

    cv2.drawContours(filled_frame, field_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    if len(weight_contours) > 0:
        if weight_contours is None:
            print("OK")
            return frame

        for w_cont in weight_contours:
            w_epsilon = 0.028 * cv2.arcLength(w_cont, True)
            w_approx = cv2.approxPolyDP(w_cont, w_epsilon, True)
            w_area = cv2.contourArea(w_cont)
            w_edge_count = len(w_approx)

            if w_area > 2500:
                if w_edge_count == 3:
                    cv2.drawContours(frame, [w_approx], -1, (255, 255, 0), 2)
                    x, y, w, h = cv2.boundingRect(w_approx)
                    color = getDominantColor(x, y, w, h, clean_frame)

                    if color == "Red":
                        cv2.drawContours(frame, [w_approx], -1, (0, 255, 0), 2)
                        print(f"Triangle : {w_area}")

                        center_x = x + w / 2
                        center_y = y + h / 1.5
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
                        cv2.putText(frame, "Triangle Target", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if w_edge_count == 4:
                    print(f"AREA: {w_area}")
                
                    x, y, w, h = cv2.boundingRect(w_approx)
                    aspect_ratio = float(w) / h

                    if 0.80 <= aspect_ratio <= 1.30:
                        print(f"Ratio: {aspect_ratio}")
                        cv2.drawContours(frame, [w_approx], -1, (0, 255, 0), 2)
                        shape_color = getDominantColor(x, y, w, h, clean_frame)
                        if shape_color == "Blue":
                            update_add_detected_shape("Square", "Blue", (x, y))
                        elif shape_color == "Red":
                            update_add_detected_shape("Square", "Red", (x, y))

                        cv2.putText(frame, f"Weight {shape_color}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if len(field_contours) > 0:
        for idx, cont in enumerate(field_contours):
            epsilon = 0.028 * cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, epsilon, True)
            area = cv2.contourArea(cont)

            if area > 2500:
                edge_count = len(approx)

                if edge_count is None or edge_count not in [3, 4, 6]:
                    continue

                if edge_count == 3 and area > 500:
                    x, y, w, h = cv2.boundingRect(approx)
                    color = getDominantColor(x, y, w, h, clean_frame)

                    if color == "Red":
                        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                        print(f"Triangle : {area}")
                        
                        add_detected_shape_queue("Triangle", "Red", (x, y))

                        center_x = x + w / 2
                        center_y = y + h / 1.5
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
                        cv2.putText(frame, "Triangle Target", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                if edge_count == 4:
                    print(f"AREA: {area}")
                    print(f"IDX: {idx}")
                
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h

                    if 0.80 <= aspect_ratio <= 1.30:
                        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                        shape_color = getDominantColor(x, y, w, h, clean_frame)
                        add_detected_shape_queue("Square", shape_color, (x, y))

                        cv2.putText(frame, f"Weight {shape_color}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if edge_count == 6 and area > 300:
                        x, y, w, h = cv2.boundingRect(approx)
                        color = getDominantColor(x, y, w, h, clean_frame)

                        if color == "Blue":
                            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                            edge_meter = np.sqrt(((h / 2) * (h / 2) + (w / 2) * (w / 2)))
                            print(f"Edge Meter: {edge_meter}")

                            calculated_ratio = w / h
                            print(f"Calculated Ratio: {calculated_ratio}")
                            
                            ratio = h * 2 / np.sqrt(3) * 1.154 * h

                            if 1.00 <= calculated_ratio <= 1.20:
                                cv2.putText(frame, "Hexagon Target", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 5, (0, 0, 255), -1)
                                add_detected_shape_queue("Hexagon", "Blue", (x, y))
    else:
        pass

    cv2.imshow("Original", frame)
    return frame

# ----------------------- DRONE MISSION PART -----------------------

class DroneMission:
    def __init__(self, connection_string='127.0.0.1:14550'):
        self.vehicle = None
        self.connection_string = connection_string
        self.waypoints = [
            (-35.3631743, 149.1653040),
            (-35.3631426, 149.1652115),
            (-35.3631836, 149.1651133),
            (-35.3632243, 149.1651129),
            (-35.3632243, 149.1651129),
            (-35.3633589, 149.1652997),
            (-35.3633928, 149.1652507),
            (-35.3633698, 149.1651460),
            (-35.3633168, 149.1651220),
            (-35.3632368, 149.1653037)
        ]
        
    def connect_vehicle(self):
        print("Connecting to vehicle...")
        self.vehicle = connect(self.connection_string, wait_ready=True)
        print("Connected to vehicle!")
        
    def arm_and_takeoff(self, target_altitude):
        print("Pre-arm checks...")
        while not self.vehicle.is_armable:
            print("Vehicle not armable, waiting...")
            time.sleep(1)

        print("Arming motors...")
        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True

        while not self.vehicle.armed:
            print("Waiting for arming...")
            time.sleep(1)

        print(f"Taking off to {target_altitude} meters...")
        self.vehicle.simple_takeoff(target_altitude)

        while True:
            print(f"Current altitude: {self.vehicle.location.global_relative_frame.alt}")
            if self.vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
                print("Target altitude reached!")
                break
            time.sleep(1)
            
    def get_distance_metres(self, location1, location2):
        dlat = location2[0] - location1[0]
        dlon = location2[1] - location1[1]
        return math.sqrt((dlat * 111139) ** 2 + (dlon * 111139 * math.cos(math.radians(location1[0]))) ** 2)

    def interpolate_waypoints(self, start, end, num_points):
        lat1, lon1 = start
        lat2, lon2 = end
        interpolated = []
        for i in range(1, num_points + 1):
            fraction = i / (num_points + 1)
            new_lat = lat1 + (lat2 - lat1) * fraction
            new_lon = lon1 + (lon2 - lon1) * fraction
            interpolated.append((new_lat, new_lon))
        return interpolated

    def goto_position(self, target_lat, target_lon, target_altitude, speed=4, distance_tolerance=1):
        print(f"Going to target position: ({target_lat}, {target_lon}, {target_altitude}m)")
        target_location = LocationGlobalRelative(target_lat, target_lon, target_altitude)
        self.vehicle.simple_goto(target_location, groundspeed=speed)

        while True:
            current_location = (self.vehicle.location.global_relative_frame.lat, self.vehicle.location.global_relative_frame.lon)
            target_location = (target_lat, target_lon)
            distance = self.get_distance_metres(current_location, target_location)
            print(f"Distance to target: {distance:.2f} metres")
            if distance < distance_tolerance:
                print("Reached target position!")
                break
            time.sleep(0.5)
            
    def set_navigation_parameters(self):
        print("Setting navigation parameters...")
        self.vehicle.parameters['WPNAV_SPEED'] = 400  
        self.vehicle.parameters['WPNAV_ACCEL'] = 50   
        self.vehicle.parameters['WPNAV_RADIUS'] = 500 

    def execute_mission(self):
        try:
            self.connect_vehicle()
            
            self.set_navigation_parameters()
            
            self.arm_and_takeoff(8)

            for i in range(len(self.waypoints)):
                print(f"\nProcessing waypoint {i+1}/{len(self.waypoints)}...")
                current_wp = self.waypoints[i]
                
                if i < len(self.waypoints) - 1:
                    next_wp = self.waypoints[i + 1]
                    distance = self.get_distance_metres(current_wp, next_wp)
                    num_intermediate = max(1, int(distance / 2))
                    intermediate_points = self.interpolate_waypoints(current_wp, next_wp, num_intermediate)
                    
                    self.goto_position(current_wp[0], current_wp[1], 8, speed=4)
                    #ehehe
                    for j, (lat, lon) in enumerate(intermediate_points, 1):
                        print(f"Processing intermediate point {j}/{num_intermediate}...")
                        self.goto_position(lat, lon, 8, speed=4)
                else:
                    self.goto_position(current_wp[0], current_wp[1], 8, speed=4)

            print("Landing...")
            self.vehicle.mode = VehicleMode("LAND")

            while self.vehicle.location.global_relative_frame.alt > 0.1:
                print(f"Current altitude: {self.vehicle.location.global_relative_frame.alt}")
                time.sleep(1)
            print("Landing complete!")

            self.vehicle.close()

        except Exception as e:
            print(f"Error occurred: {e}")
            if self.vehicle:
                self.vehicle.mode = VehicleMode("LAND")  
                self.vehicle.close()

# ----------------------- MAIN FUNCTION -----------------------

def main():
    if not rospy.core.is_initialized():
        rospy.init_node('gazebo_mission', anonymous=True)
    
    camera = CameraSubscriber()
    
    listener_thread = threading.Thread(target=process_detected_shapes, daemon=True)
    listener_thread.start()
    
    cleaner_thread = threading.Thread(target=remove_old_shapes, daemon=True)
    cleaner_thread.start()
    
    drone_mission = DroneMission()
    mission_thread = threading.Thread(target=drone_mission.execute_mission, daemon=True)
    mission_thread.start()
    
    print("Starting image processing...")
    
    try:
        while not rospy.is_shutdown():
            if camera.latest_frame is not None:
                frame = camera.latest_frame.copy()
                processed_frame = process_frame(frame)
                
                if detected_shapes:
                    print("\n🔵 Shapes on screen:")
                    for shape_id, shape in detected_shapes.items():
                        print(f"  ➜ {shape['color']} {shape['type']} - Position: {shape['position']}")
                else:
                    print("\n⚠️ No shapes on screen!")
                
                if initalizeKeyBinds():
                    break
                
                time.sleep(0.05)  

    except KeyboardInterrupt:
        print("Shutting down...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()