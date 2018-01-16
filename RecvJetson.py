import socket
import os
import sys
import time

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost',9595))

while True:
    s.listen(5)
    leave = False

    conn, addr = s.accept()
    l = 0
    print('Connected to '+str(addr))
    while not leave:
        incoming = conn.recv(1000)
        if incoming=='':
            continue
        for command in incoming.split('~:~'):
            if(len(command)==0):
                continue
            if(command[0]=='$'):
                print(command[1:])
                op = os.open('/home/ayush/ball_detect_cnn/1'+command[1:],os.O_CREAT)
                op = os.open('/home/ayush/ball_detect_cnn/1'+command[1:],os.O_RDWR)
                print('yolo')
            if(command[0]=='%'):
                # print(sys.getsizeof(command))
                k=os.write(op,command[1:])
                print(k)
            if(command[0]=='^'):
                print('Closing File: '+str(os.close(op)))
            if(command[0]=='*'):
                leave = True
                break
    conn.close()
    print('Connection closed')