# import socket  # 导入 socket 模块
#
# s = socket.socket()  # 创建 socket 对象
# host = "127.0.0.1"  # 获取本地主机名
# port = 8080  # 设置端口号
#
# s.connect((host, port))
# print(s.recv(1024).decode("utf-8"))
#
# s.close()
import time
import winsound


winsound.Beep(500,500)
print(time.time())





