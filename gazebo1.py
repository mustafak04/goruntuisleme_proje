import cv2
import numpy as np
import time
import queue
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy

SHAPE_TIMEOUT = 0.5

#Queue for detected shapes
shape_queue = queue.Queue()
#Dictionary for detected shapes
detected_shapes = {}

class CameraSubscriber:
    def __init__(self):
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
        "type":shape_type,
        "color":shape_color,
        "position":position,
        "time_stamp":time.time()
    }
    shape_queue.put(shape_info)

def remove_old_shapes():
    while True:
        time.sleep(1)
        current_time = time.time()
        to_remove = [key for key, shape in detected_shapes.items() if current_time - shape["time_stamp"] > SHAPE_TIMEOUT]

        for key in to_remove:
            del detected_shapes[key]
        # cv2.waitKey(1)

def update_add_detected_shape(shape_type, shape_color, shape_position):
    shape_id = f"{shape_type}_{shape_color}"
    shape_info = {
        "id":shape_id,
        "type":shape_type,
        "color":shape_color,
        "position":shape_position,
        "time_stamp":time.time()
    }
    detected_shapes[shape_id] = shape_info

def process_detected_shapes():
    while True:
        shape_info = shape_queue.get()
        print(f"Detected Objects {shape_info['type']}")
        update_add_detected_shape(
            shape_type=shape_info["type"],
            shape_color=shape_info["color"],
            shape_position=shape_info['position']
        )

listener_thread = threading.Thread(target=process_detected_shapes,daemon=True)
listener_thread.start()

cleaner_thread = threading.Thread(target=remove_old_shapes,daemon=True)
cleaner_thread.start()


#capture the video with given path as video_path
video = cv2.VideoCapture(0)

#test

def checkVideoStartState():
    if not video.isOpened():
        print("Error: Could not open video.")
        exit()

checkVideoStartState()

def initalizeKeyBinds()->bool:  
    #keybind is a variable that holds the key value that you pressed ('q')
    keybind = cv2.waitKey(1) &0xFF 
    if keybind == ord('q'):
        #cv2.destroyAllWindows() is used to close the window              
        cv2.destroyAllWindows()
        return True
    return False

cam2 = CameraSubscriber()
#rospy.spin()

#gets frame width of video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#gets frame height of video
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#get frame count of video
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#get fps of video
fps = int(video.get(cv2.CAP_PROP_FPS))

#show video properties when soft started
print("Video Properties")
print("----------------------")
#print video width
print("Width: ",frame_width)
#print video height
print("Height: ",frame_height)
#print video total frame count
print("Frame Count: ",frame_count)
#print video fps (one time)
print("FPS: ",fps)
print("----------------------")

#method for passing
def nothing():
    pass

def getDominantColor(x,y,w,h):
    shape_colors:list = ('b','g','r')
    shape_dominant_index = None
    histograms = {}
    shape_dominant_color = {}

    color_names = {
    'r': "Red",
    'g': "Green",
    'b': "Blue",
    }

    crop_image = clean_frame[y:y+h,x:x+w]

    for i,color in enumerate(shape_colors):
        color_hist = cv2.calcHist([crop_image],[i],None,[256],[0,256])
        histograms[color] = color_hist
    for color in histograms:
        max_intensity = np.argmax(histograms[color])
        shape_dominant_color[color] = max_intensity
    shape_dominant_index = max(shape_dominant_color,key=shape_dominant_color.get)
    return color_names[shape_dominant_index]

def getBorderDominantColor(x, y, w, h, approx):
    border_dominant_color = {}
    histograms = {}
    border_colors = ('b', 'g', 'r')
    border_dominant_index = None
    histograms = {}
    border_dominant_color = {}
    color_names = {
        'r': "Red",
        'g': "Green",
        'b': "Blue",
    }

    crop_image_outline = clean_frame[y:h + y, x:w + x]

    border_width = 100  # Kenar bandı genişliği
    outline = clean_frame[y:y + h, x:x + w] 

    mask = np.zeros_like(outline, dtype=np.uint8)
    cv2.drawContours(mask, [approx - [x, y]], -1, (255, 255, 255), thickness=border_width)  # Kenar kalınlığı
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

def get_full_screen_dominant_color():
    _frame = frame.copy()
    # BGR kanalları
    channels = ('b', 'g', 'r')
    color_names = {'r': "Red", 'g': "Green", 'b': "Blue"}
    color_ratios = {}

    # Toplam piksel sayısı
    total_pixels = _frame.shape[0] * _frame.shape[1]

    # Renk kanallarını al
    b_channel, g_channel, r_channel = cv2.split(_frame)
    
    # Her renk kanalının yoğunluğunu hesapla
    b_intensity = np.sum(b_channel) / (total_pixels * 255) * 100  # % cinsinden
    g_intensity = np.sum(g_channel) / (total_pixels * 255) * 100
    r_intensity = np.sum(r_channel) / (total_pixels * 255) * 100

    general_intensity = (b_intensity + g_intensity + r_intensity) / 3

    # Yüzde oranlarını kaydet
    color_ratios['b'] = b_intensity
    color_ratios['g'] = g_intensity
    color_ratios['r'] = r_intensity

    # Yüzde oranlarını yazdır
    print(f"🔵 Blue: {color_ratios['b']:.2f}% | 🟢 Green: {color_ratios['g']:.2f}% | 🔴 Red: {color_ratios['r']:.2f}%")

    # Renklerin %50'yi geçip geçmediğini kontrol et
    dominant_colors = [color for color in color_ratios if color_ratios[color] > 30]
    
    if dominant_colors:
        dominant_color = dominant_colors[0]
        dominant_color_intensity = color_ratios[dominant_color]
        if abs(dominant_color_intensity - general_intensity >= 10):
            print(abs(dominant_color_intensity - general_intensity))
            print(abs(dominant_color_intensity - general_intensity))
            
            return color_names[dominant_colors[0]]

    return "No dominant color"  # %50'yi geçmeyqen bir durum varsa

while not rospy.is_shutdown():
    if cam2.latest_frame is not None:
        frame = cam2.latest_frame.copy()

        if detected_shapes:
            print("\n🔵 Şu an ekrandaki şekiller:")
            for shape_id, shape in detected_shapes.items():
                print(f"  ➜ {shape['color']} {shape['type']} - Pozisyon: {shape['position']}")
        else:
            print("\n⚠️ Hiçbir şekil ekranda değil!")
        cv2.waitKey(1)
        #----------- HSV CONFIG ------------
        clean_frame = frame.copy()
        hsv_image = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        # Mavi renk aralığı
        blue_lower = np.array([95, 100, 50])   # Mavi renk için alt sınır
        blue_upper = np.array([140, 255, 255])  # Mavi renk için üst sınır

        # Kırmızı renk aralığı
        red_lower_1 = np.array([-2, 120, 50])     # Kırmızı renk için alt sınır 1
        red_upper_1 = np.array([12, 255, 255])   # Kırmızı renk için üst sınır 1

        red_lower_2 = np.array([165, 120, 50])   # Kırmızı renk için alt sınır 2
        red_upper_2 = np.array([185, 255, 255])  # Kırmızı renk için üst sınır 2

        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

        # Kırmızı için maske oluşturun (iki aralık)
        red_mask_1 = cv2.inRange(hsv_image, red_lower_1, red_upper_1)
        red_mask_2 = cv2.inRange(hsv_image, red_lower_2, red_upper_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        overlap_mask = cv2.bitwise_and(blue_mask, red_mask)

        final_red = cv2.subtract(red_mask, overlap_mask)
        final_blue = cv2.subtract(blue_mask, overlap_mask)

        final_mask = cv2.bitwise_or(final_red, final_blue)
        cv2.imshow("Final Mask",final_mask)

        hsv_result = cv2.bitwise_and(frame, frame, mask=final_mask)
        
        gray_result = cv2.cvtColor(hsv_result, cv2.COLOR_BGR2GRAY)

        gray_result = cv2.medianBlur(gray_result, 7)
        gray_result = cv2.erode(gray_result, kernel=np.ones((5, 5), dtype=np.uint8), iterations=3)
        gray_result = cv2.morphologyEx(gray_result, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.uint8), iterations=3)

        #---------------- GRAY CONFIG ---------------
        clean2_frame = frame.copy()

        gray_frame = cv2.cvtColor(clean2_frame,cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        equalized_frame = cv2.equalizeHist(blurred_frame)
        _, otsu_thresh = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thresh_val, _ = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        w_edges = cv2.Canny(blurred_frame, otsu_thresh_val * 0.5, otsu_thresh_val)
        fields = cv2.Canny(gray_result, 100, 200)

        weight_contours,_ = cv2.findContours(w_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        field_contours,_ = cv2.findContours(fields,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        filled_frame = np.zeros_like(fields)

        cv2.drawContours(filled_frame,field_contours,-1,(255,255,255),thickness=cv2.FILLED)
        cv2.imshow("Second Frame",w_edges)
        #cv2.imshow("Weight Frame",weight_frame)
        #cv2.imshow("Filled Frame",filled_frame)

        full_screen_intense = get_full_screen_dominant_color()
        print("Full Screen Intense: ",full_screen_intense)

        if(full_screen_intense == "Blue"):
            update_add_detected_shape("Hexagon","Blue",(0,0))
        elif (full_screen_intense == "Red"):
            update_add_detected_shape("Triangle","Red",(0,0))

        if len(weight_contours) > 0:
            if (weight_contours is None):
                print("OK")
                break

            for w_cont in weight_contours:
                w_epsilon = 0.028*cv2.arcLength(w_cont,True)
                w_approx = cv2.approxPolyDP(w_cont,w_epsilon,True)
                w_area = cv2.contourArea(w_cont)
                w_edge_count = len(w_approx)

                if(w_area > 2500):
                    if(w_edge_count == 3):
                        cv2.drawContours(frame,[w_approx],-1,(255,255,0),2)

                        x,y,w,h = cv2.boundingRect(w_approx)
                        color = getDominantColor(x,y,w,h)

                        if(color == "Red"):
                            cv2.drawContours(frame, [w_approx], -1, (0, 255, 0), 2)
                            print(f"Triangle : {w_area}")

                            center_x = x+w/2
                            center_y = y+h/1.5
                            cv2.circle(frame,(int(center_x),int(center_y)),5,(255,0,0),-1)
                            cv2.putText(frame,"Triangle Target",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                    if(w_edge_count==4):
                        print(f"AREA: {w_area}")
                    
                        x,y,w,h = cv2.boundingRect(w_approx)

                        aspect_ratio = float(w)/h

                        if (0.80<= aspect_ratio <= 1.30):
                            print(f"Ratio: {aspect_ratio}")
                            cv2.drawContours(frame, [w_approx], -1, (0, 255, 0), 2)
                            shape_color = getDominantColor(x,y,w,h)
                            if(shape_color == "Blue"):
                                update_add_detected_shape("Square","Blue",(x,y))
                            elif (shape_color == "Red"):
                                update_add_detected_shape("Square","Red",(x,y))
                            cv2.putText(frame,f"Weight {shape_color}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        if len(field_contours) > 0:

            for idx,cont in enumerate(field_contours):
                epsilon = 0.028*cv2.arcLength(cont,True)
                approx = cv2.approxPolyDP(cont,epsilon,True)
                area = cv2.contourArea(cont)

                if area > 2500:
                    edge_count = len(approx)

                    if(edge_count is None or edge_count != 3 and edge_count != 4 and edge_count != 6):
                        continue
                    else:
                        pass

                    if(edge_count==3) and (area > 500):
                        x,y,w,h = cv2.boundingRect(approx)

                        color = getDominantColor(x,y,w,h)

                        if(color == "Red"):
                            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                            print(f"Triangle : {area}")
                            
                            add_detected_shape_queue("Triangle","Red",(x,y))

                            center_x = x+w/2
                            center_y = y+h/1.5
                            cv2.circle(frame,(int(center_x),int(center_y)),5,(255,0,0),-1)
                            cv2.putText(frame,"Triangle Target",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                    if(edge_count==4):
                        print(f"AREA: {area}")
                        print(f"IDX: {idx}")

                        x,y,w,h = cv2.boundingRect(approx)

                        aspect_ratio = float(w)/h

                        if (0.80<= aspect_ratio <= 1.30):
                            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

                            shape_color = getDominantColor(x,y,w,h)
                            add_detected_shape_queue("Square",shape_color,(x,y))

                            cv2.putText(frame,f"Weight {shape_color}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

                    if(edge_count==6) and (area >300):
                            x,y,w,h = cv2.boundingRect(approx)

                            color = getDominantColor(x,y,w,h)

                            if(color == "Blue"):
                                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

                                edge_meter = np.sqrt(((h/2)*(h/2) + (w/2)*(w/2)))
                                print(f"Edge Meter: {edge_meter}")

                                calculated_ratio = w/h
                                print(f"Calculated Ratio: {calculated_ratio}")

                                ratio= h*2/np.sqrt(3) *1.154 *h

                                if (1.00 <= calculated_ratio <= 1.20):
                                    cv2.putText(frame,"Hexagon Target",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

                                    cv2.circle(frame,(int(x+w/2),int(y+h/2)),5,(0,0,255),-1)
                                    add_detected_shape_queue("Hexagon","Blue",(x,y))
        else:
            pass

        cv2.imshow("Original",frame)

        if initalizeKeyBinds():
            break