import socket
import os
import shutil
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('10.42.0.94',9596))
while True:
    s.listen(5)
    leave = False

    conn, addr = s.accept()
    l = 0
    print('Connected to '+str(addr))
    while not leave:
        incoming = conn.recv(1024)
        if incoming=='':
            continue
        for command in incoming.split('~:~'):
            if(len(command)==0):
                continue
            if(command[0]=='!'):
                os.chdir(command[1:])
                main_direct = command[1:]
            if(command[0]=='@'):
                if(os.path.isdir(main_direct+'/'+command[1:])):
                    # shutil.rmtree(main_direct+'/'+command[1:])
                    pass
                print(command[1:])
                print(os.path.isdir(main_direct+'/'+command[1:]))
            if(command[0]=='#'):
                print(command[1:])
                # os.mkdir(command[1:])
            if(command[0]=='$'):
                print(command[1:])
                op = os.open('/home/ayush/ball_detect_cnn/in.jpg',os.O_CREAT)
                op = os.open('/home/ayush/ball_detect_cnn/in.jpg',os.O_RDWR)
                print('yolo')
            if(command[0]=='%'):
                print(sys.getsizeof(command))
                k=os.write(op,command[1:])
                print(k)
            if(command[0]=='^'):
                print('Closing File: '+str(os.close(op)))
            if(command[0]=='&'):
                op = os.open(main_direct+'/'+command[1:],os.O_RDWR)
                t = os.read(op,1023)
                while(len(t)!=0):
                    print(conn.send('%'+t))
                    t = os.read(op,1023)
                conn.send('~:~^')
                os.close(op)
                leave = True
                break
            if(command[0]=='*'):
                leave = True
                break
    conn.close()
    print('Connection closed')