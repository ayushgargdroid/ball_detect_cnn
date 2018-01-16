import socket
import matplotlib.pyplot as plt

def validate(s):
    first_occurence = s.find('lat')
    if s.find('lat',first_occurence+3) is not -1:
        return False
    first_occurence = s.find('lon')
    if s.find('lon',first_occurence+3) is not -1:
        return False
    if len(s) < 20:
        return False
    return True

def distance(y1,x1,y2,x2):
    # print('distttt',y1,x1)
    y=(y2-y1)*100
    x=(x2-x1)*100
    z=x**2+y**2
    d=(pow(z,0.5))*1000
    # print('DISTANCE=',d)
    return d


s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(('192.168.43.113',5051))
lats = []
lons = []
starting = True
# fig = plt.figure(figsize=(7,5))
# ax = fig.add_axes([0.15,0.15,0.81,0.81])
# ax.set_ylim(13,14)
# ax.set_xlim(71,75)
plt.axis([13.3470,13.3484,74.7916,74.7922])
plt.ion()

while True:
    incoming_msg = s.recv(72)
    # print(incoming_msg)
    if not validate(incoming_msg):
        continue
    first_occurence_lat = incoming_msg.find('lat')
    first_occurence_lon = incoming_msg.find('lon')
    latitude = incoming_msg[first_occurence_lat+3:first_occurence_lon-4]
    longitude = incoming_msg[first_occurence_lon+3:]
    longitude = longitude[:len(longitude)-4]
    print('Lat: '+str(latitude)+' Lon: '+str(longitude))
    latitude = float(latitude)
    longitude = float(longitude)
    if len(lats) >= 50:
        lats = lats[1:]
    if len(lons) >= 50:
        lons = lons[1:]
    if len(lats) > 10:
        # total_dist = 0
        # for i in range(-1,-11,-1):
        #     total_dist = total_dist + distance(lats[i],lons[i],lats[i-1],lons[i-1])
        # avg_dist = total_dist/10
        if distance(lats[-1],lons[-1],latitude,longitude) > 10:
            print 'Skipped'
            continue
    if not starting:
        plt.plot(lats,lons,marker='*',color='blue')
        plt.plot([lats[-1],latitude],[lons[-1],longitude],marker='*',color='red')
    if starting:
        starting = False
        plt.plot([latitude],[longitude],marker='*',color='blue')
    lats.append(latitude)
    lons.append(longitude)
    plt.pause(0.1)


'''
from gps3 import gps3
import socket
import sys
gps_socket = gps3.GPSDSocket()
data_stream = gps3.DataStream()
gps_socket.connect()
gps_socket.watch()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('192.168.43.113',5050))
while True:
    s.listen(1)
    conn,addr = s.accept()
    try:
        for new_data in gps_socket:
            if new_data:
                data_stream.unpack(new_data)
                # print('Altitude = ', data_stream.TPV['alt'])
                print('Latitude = ', data_stream.TPV['lat'])
                print('longitude= ',data_stream.TPV['lon'])
                print(type(data_stream.TPV['lon']))
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                conn.send('lat'+str(data_stream.TPV['lat'])+'lon'+str(data_stream.TPV['lon']))
    except KeyboardInterrupt:
        conn.close()
        print 'Exiting program'
        break
'''