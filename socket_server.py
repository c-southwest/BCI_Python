import socket
import time

s = socket.socket()
host = "127.0.0.1"
port = 8080
s.bind((host, port))
s.listen(5)
client, address = s.accept()
while True:
    a = "1"
    client.send(a.encode("utf-8"))
    print(a)
    time.sleep(1)
client.close()