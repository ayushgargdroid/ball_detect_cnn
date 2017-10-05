import socket
import os
import sys
import time
#'C:\\Users\\MRM\\Ayush\\ball_detect_cnn\\my_model11.h5'

def recur(direct):
    global directories, directi, directo
    current = os.getcwd()
    os.chdir(direct)
    subs = os.listdir(os.curdir)
    for i in subs:
        if(os.path.isdir(i)):
            directories += '~:~#'+os.getcwd()+'/'+i
            recur(i)
            os.chdir(current+'/'+direct)
        elif('.jpg' in i):
            temp = ('$'+os.getcwd()+'/'+i).replace(directi, directo)
            print(temp)
            s.send(temp)
            time.sleep(0.1)
            op = os.open(os.getcwd()+'/'+i,os.O_RDONLY)
            t = os.read(op,1023)
            l=0
            while(len(t)!=0):
                print(s.send('%'+t))
                # print(t.encode('string_escape'))
                t = os.read(op,1023)
                l+=1
            s.send('~:~^')
            time.sleep(0.1)
            print(l)
            os.close(op)

def main():
    global directories, directi, directo
    choice = input('1.Upload 2.Download 3.Exit\n')
    if(choice==1):
        directi = input('Directory at the source:\n')
        directo = input('Directory at the destination:\n')
        if(os.path.isdir(directi)):
            s1 = '!'+directo[:directo.rfind('\\')]
            s2 = '@'+directo[directo.rfind('\\')+1:]
            s.send(s1+'~:~'+s2)
            os.chdir(directi[:directi.rfind('/')])
            recur(directi[directi.rfind('/')+1:])
            directories = directories[1:]
            directories = directories.replace(directi,directo)
            print directories
            s.send(directories)
            s.send('~:~*')
            s.close()
            print('Connection closed')
            # os.chdir(direct[:direct.rfind('/')])
            #recur(direct[direct.rfind('/')+1:])
    
    elif(choice==2):
        directo = input('Directory at the source:\n')
        directi = input('Directory at the destination:\n')
        leave = False
        if(not os.path.isdir(directi)):
            s1 = '!'+directo[:directo.rfind('\\')]
            s2 = '&'+directo[directo.rfind('\\')+1:]
            s.send(s1+'~:~'+s2)
            op = os.open(directi,os.O_CREAT)
            op = os.open(directi,os.O_RDWR)
            while not leave:
                incoming = s.recv(1024)
                if incoming=='':
                    continue
                for command in incoming.split('~:~'):
                    if(len(command)==0):
                        continue
                    if(command[0]=='%'):
                        os.write(op,command[1:])
                    if(command[0]=='^'):
                        os.close(op)
                        print('File written')
                        leave = True
                        break
            s.close()
            print('Connection closed')
    else:
        s.send('~:~*')
        s.close()
        sys.exit(0)

while True:
    directories = ''
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect(('10.42.0.94',9596))
    main()