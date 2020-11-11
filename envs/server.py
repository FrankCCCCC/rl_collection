# Echo server program
import socket
import json

MAXLEN = 1024
ENCODE_FORMAT = 'UTF-8'
HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 50007              # Arbitrary non-privileged port

def encode(data):
    # Convert dict object to bytes in JSON format
    # return json.dumps(data).encode(encoding = ENCODE_FORMAT)
    return json.dumps(data).encode()

def decode(data):
    # Convert bytes in JSON format to dict object
    # return json.loads(data.decode(encoding = ENCODE_FORMAT))
    return json.loads(data.decode())

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)

conn, addr = s.accept()
print('Connected by', addr)
while 1:
    recv_msg = conn.recv(MAXLEN)
    if recv_msg != b'':
        res = decode(recv_msg)

        print(res)
        if res['request'] == 'init':
            print(f'Request: init')
            conn.send(encode({'state_size': 4, 'action_size': 2}))
        elif res['request'] == 'reset':
            print(f'Request: reset')
            conn.send(encode({'state': [1, 2, 3, 4]}))
        elif res['request'] == 'act':
            print(f'Request: act, Action: {res["action"]}')
            conn.send(encode({'state': [2, 4, 6, 8], 'reward': 5, 'is_done': True, 'info': None}))
    else:
        conn.close()
        conn, addr = s.accept()
        print('Connected by', addr)
    
    
#   conn.sendall(data)
