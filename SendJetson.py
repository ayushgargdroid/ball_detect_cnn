import socket
import os
import sys
import time

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(('localhost',9595))

directory = input('Enter file path')
op = os.open(directory,os.O_RDONLY)
name = directory[directory.rfind('/')+1:]
s.send('~:~$'+name)
t = os.read(op,1023)
l=0
time.sleep(0.1)
# s.send('')
while(len(t)!=0):
    print(s.send('~:~%'+t))
    t = os.read(op,1000)
    l+=1
    time.sleep(0.1)
s.send('~:~^')
time.sleep(0.1)
print(l)
os.close(op)
s.send('~:~*')
s.close()