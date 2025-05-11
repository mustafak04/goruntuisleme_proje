from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math

vehicle = connect('127.0.0.1:14550', wait_ready=True)

# Koordinat listesi (enlem, boylam)
waypoints = [
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
def arm_and_takeoff(target_altitude):
    print("Pre-arm kontrolleri yapılıyor...")
    while not vehicle.is_armable:
        print("Araç arm edilebilir durumda değil, bekleniyor...")
        time.sleep(1)

    print("Motorlar arm ediliyor...")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Arm işlemi bekleniyor...")
        time.sleep(1)

    print(f"{target_altitude} metre yüksekliğe kalkış yapılıyor...")
    vehicle.simple_takeoff(target_altitude)

    while True:
        print(f"Mevcut yükseklik: {vehicle.location.global_relative_frame.alt}")
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            print("Hedef yüksekliğe ulaşıldı!")
            break
        time.sleep(1)

def get_distance_metres(location1, location2):
    """
    İki konum arasındaki mesafeyi metre cinsinden hesaplar.
    location1 ve location2: (lat, lon) tuple'ları
    """
    dlat = location2[0] - location1[0]
    dlon = location2[1] - location1[1]
    return math.sqrt((dlat * 111139) ** 2 + (dlon * 111139 * math.cos(math.radians(location1[0]))) ** 2)

def interpolate_waypoints(start, end, num_points):
    """
    İki waypoint arasında ara noktalar oluşturur.
    start, end: (lat, lon) tuple'ları
    num_points: Ara nokta sayısı
    """
    lat1, lon1 = start
    lat2, lon2 = end
    interpolated = []
    for i in range(1, num_points + 1):
        fraction = i / (num_points + 1)
        new_lat = lat1 + (lat2 - lat1) * fraction
        new_lon = lon1 + (lon2 - lon1) * fraction
        interpolated.append((new_lat, new_lon))
    return interpolated

def goto_position(target_lat, target_lon, target_altitude, speed=2, distance_tolerance=1):
    """
    Drone'u belirtilen koordinatlara gönderir.
    """
    print(f"Hedef konuma gidiliyor: ({target_lat}, {target_lon}, {target_altitude}m)")
    target_location = LocationGlobalRelative(target_lat, target_lon, target_altitude)
    vehicle.simple_goto(target_location, groundspeed=speed)

    while True:
        current_location = (vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon)
        target_location = (target_lat, target_lon)
        distance = get_distance_metres(current_location, target_location)
        print(f"Hedefe kalan mesafe: {distance:.2f} metre")
        if distance < distance_tolerance:
            print("Hedef konuma ulaşıldı!")
            break
        time.sleep(0.5)

# Navigasyon parametrelerini ayarla
print("Navigasyon parametreleri ayarlanıyor...")
vehicle.parameters['WPNAV_SPEED'] = 200  # 2 m/s
vehicle.parameters['WPNAV_ACCEL'] = 50   # 50 cm/s²
vehicle.parameters['WPNAV_RADIUS'] = 500 # 5 metre

# Görev yürütme
try:
    # 1. Arm ve 5 metre kalkış
    arm_and_takeoff(5)

    # 2. Koordinatları sırasıyla gez
    for i in range(len(waypoints)):
        print(f"\nWaypoint {i+1}/{len(waypoints)} işleniyor...")
        current_wp = waypoints[i]
        
        # Eğer bir sonraki waypoint varsa, ara noktalar oluştur
        if i < len(waypoints) - 1:
            next_wp = waypoints[i + 1]
            # İki waypoint arasındaki mesafeyi hesapla
            distance = get_distance_metres(current_wp, next_wp)
            # Mesafeye bağlı olarak ara nokta sayısı belirle (örneğin, her 2 metrede bir)
            num_intermediate = max(1, int(distance / 2))
            intermediate_points = interpolate_waypoints(current_wp, next_wp, num_intermediate)
            
            # Mevcut waypoint'a git
            goto_position(current_wp[0], current_wp[1], 5, speed=2)
            
            # Ara noktaları gez
            for j, (lat, lon) in enumerate(intermediate_points, 1):
                print(f"Ara nokta {j}/{num_intermediate} işleniyor...")
                goto_position(lat, lon, 5, speed=2)
        else:
            # Son waypoint'a git
            goto_position(current_wp[0], current_wp[1], 5, speed=2)

    # 3. Son konumda iniş
    print("İniş yapılıyor...")
    vehicle.mode = VehicleMode("LAND")

    # İniş tamamlanana kadar bekle
    while vehicle.location.global_relative_frame.alt > 0.1:
        print(f"Mevcut yükseklik: {vehicle.location.global_relative_frame.alt}")
        time.sleep(1)
    print("İniş tamamlandı!")

    # 4. Bağlantıyı kapat
    vehicle.close()

except Exception as e:
    print(f"Hata oluştu: {e}")
    vehicle.mode = VehicleMode("LAND")  # Hata durumunda iniş
    vehicle.close()