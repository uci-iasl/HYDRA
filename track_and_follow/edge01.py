import threading
import _thread
import time
import socket
from devices.blocks import Logger, Module, ConsumeImageProduceFeatDet, ConsumeFeatProduceAction, PosSender
from devices.FakeDrone import FakeDrone
from queue import Queue as Queue
from queue import Empty
import pickle
from devices.config import *
from devices import net_p3
import argparse
from influxdb import InfluxDBClient


ADAPTIVE_PIPELINES = False

parser = argparse.ArgumentParser()
parser.add_argument("ip")
parser.add_argument("--name", help="", default="EDGE01")
parser.add_argument("--model", type=int, default=0)
parser.add_argument("--db", type=str, help="Specify host address at which influx is running", default=None)
parser.add_argument("--trt", action="store_true", help="Is your device TRT enabled?")

# parser.add_argument("port")
args = parser.parse_args()
TCP_EDGE=args.ip
global TRT
TRT = args.trt


class EdgeServer(threading.Thread):
  def __init__(self, iid, db_host=args.db):
    threading.Thread.__init__(self)
    self.id = iid
    self.start_time = time.time()
    self.modules = {}
    self.queues = {"img": Queue(maxsize=1), "feat": Queue(maxsize=1), "action": Queue(maxsize=1)}
    self.state = {"info": Logger(name="EDGE_timing", category="flux_point", start=self.start_time,
      cats=["id", "from/to", "type", "auth"]),
                  "dev_id": self.id,
                  "is_running": True,
                  "start_time": self.start_time,
                  "active_pipelines": {"": True},
                  "status": "init",
                  "conn": {},
                  "net_interface":[TCP_EDGE],
                  "model": models[args.model]}
    # self.modules["camera"] = ImageProducer(out_q=[self.queues["img"], ], state=self.state)
    self.modules["img2feat"] = ConsumeImageProduceFeatDet(in_q=self.queues["img"],
                      out_q=[self.queues["feat"], ], state=self.state, pipe="", save_img = True)
    self.modules["feat2act"] = ConsumeFeatProduceAction(in_q=self.queues["feat"],
                      out_q=[self.queues["action"], ], state=self.state, pipe="")
    self.discoverable_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet, UDP
    self.discoverable_socket.bind(('', UDP_DISCOVERY_PORT))
    self.next_port = INITIAL_TCP_PORT
    self.modules["pos_sender"] = PosSender(state=self.state)
    if db_host is not None:
      self.state["client"] = InfluxDBClient(host=db_host, port=8086, username='admin', password='admin')

    self.state["drone"] = FakeDrone()
    self.my_tcp_ip = self.state["net_interface"][0]

  def start(self):
    self.modules["img2feat"].setDaemon(DAEMONIAC_THREADS)
    self.modules["img2feat"].start()
    time.sleep(10)
    for t_name in self.modules:
      if t_name != "img2feat":
        self.modules[t_name].setDaemon(DAEMONIAC_THREADS)
        self.modules[t_name].start()
        print("Started " + t_name)
    self.run()

  # Welcoming new connections
  def run(self):
    while True:
      print([(e, self.state["conn"]) for e in self.state["conn"]])
      data, client_addr = self.discoverable_socket.recvfrom(BUFF_SIZE) # buffer size is 1024 bytes
      if len(data) > 1:
        print(data, client_addr)
        dev_id = pickle.loads(data).split('-')[1]
        print("____")
        print(dev_id)
        print("self.state[conn]", self.state["conn"])
        if dev_id in self.state["conn"]:
          print("self.state[conn][dev_id]", self.state["conn"][dev_id])
        print("____")
        if dev_id not in self.state["conn"] or not self.state["conn"][dev_id]: # connection not in connections or not active
          # self.info.add("welcoming", ("received message: {}".format(pickle.loads(data))))
          # self.info.add(category="action2take", fields={"id": action2take.t_frame, "from/to": "cons_action", "type": "action", "auth": self.dev_id})
          print("Sending my info to Drone")
          self.discoverable_socket.sendto(pickle.dumps({"name":self.id, "ip": self.my_tcp_ip, "port": self.next_port}), client_addr)
          # self.info.add("welcoming", "{};{}:{} sent to {}:{}".format(self.id, self.my_tcp_ip,self.next_port, *client_addr))
          print("Starting a new service for {} on port {}".format(data, self.next_port, client_addr))
          e = _Service(ip=self.my_tcp_ip, port=self.next_port, state=self.state, client_addr=client_addr,
                  out_q=[self.queues["img"], ], in_q=self.queues["action"], dev_id=dev_id)
          self.next_port += 1
          print("Ready to start service")
          e.start()
          print("Service started")
      else:
        time.sleep(0.1)
      data = ''

class _Service(Module):
  def __init__(self, in_q, out_q, state, ip='', port=10000, client_addr=None, dev_id="UAV01"):
    Module.__init__(self, in_q=in_q, out_q=out_q, state=state, pipe=client_addr, name="SERVICE")
    # Create a TCP/IP socket
    self.dev_id = dev_id
    print("line 107")
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("line 109")
    self.client_addr = client_addr
    # Bind the socket to the address given on the command line
    server_address = (ip, port)
    print("Binding {}:{}".format(*server_address))
    self.sock.bind(server_address)
    print('Binded {}:{}'.format(*self.sock.getsockname()))
    self.sock.listen(1)
    # print("Listening")
    self.connection = None
    _thread.start_new_thread(self.ready_to_reply, ())
    print("Finished init")
    self.last_put = {}

  def run(self):
    rem = b""
    while True: #repeat after every accepter connection
      print('waiting for a connection')
      self.connection, self.client_address = self.sock.accept()
      self.state["conn"][self.dev_id] = True
      print('Got the connection')
      try:
        print('client connected:', self.client_address)
        while self.state["conn"][self.dev_id]:
          print("receiving")
          # data = self.connection.recv(BUFF_SIZE)
          print("before receive fat message")
          t1, t2, data, rem = net_p3.receive_fat_msg(self.connection, rem)
          print("after receive fat message")
          # self.info.add("edge_conn", 'received {!r} from {}'.format(data, self.connection.getpeername()))
          if data:
            img_decoded = pickle.loads(data)
            # print(time.time(), "MSG RECEIVED ", img_decoded.t_frame)
            self.info.add(category="put_out_q", fields={"id": img_decoded.t_frame, "from/to": "service", "type": "img", "auth": self.dev_id})
            for q in self.out_q:
              if q not in self.last_put:
                self.last_put[q] = -1
              item = None
              to_put = img_decoded
              try:
                item = q.get(False)
                if item:
                  # print("DISCARD ")
                  q.task_done()
                  if to_put.t_frame < item.t_frame:
                    to_put = item
              except Exception as e:
                # raise e
                pass
                # print("EXCEPT 119")
              finally:
                if to_put.t_frame > self.last_put[q]:
                  q.put(to_put)
                  if item is not None:
                    print("DISCARD ")
                  self.last_put[q] = to_put.t_frame
                else:
                  # print(time.time(), self.last_put , to_put.t_frame)
                  pass
          else:
            continue
      except Exception as e:
        pass
      finally:
        self.state["conn"][self.dev_id] = False
        print("Closing connection with {}".format(self.connection.getpeername()))
        self.connection.close()
        return

  def ready_to_reply(self):
    print("Ready to reply is running")
    while self.is_running:
      try:
        # print("WAITING get in SERVICE")
        action = self.in_q.get(timeout=Q_READ_TIMEOUT)
        # print("GET in ready_to_reply ", action.nice_str())
        # print("GOT in SERVICE")
      except Empty as e:
        # print(self.getName() + " - Timeout expired")
        # print(e)
        # print("EXCEPT148")
        pass
        # self.is_running = False
      else:
        self.in_q.task_done()
        tmp = pickle.dumps(action)
        # print("TMPS", tmp)
        # print(time.time(), "ACTION151: ", action.nice_str())

        # self.info.add(category="get_q", fields={"id": action.t_frame, "from/to": "replying", "type": "action", "auth": self.dev_id})
        # print(time.time(), "CONN:", self.connection)
        # while self.connection is None:
        #   pass
        # while self.connection is not None:
        # print(time.time(), "replying", action.t_frame)
        net_p3.send_fat_msg(self.connection, tmp, 0, 0)




if __name__=="__main__":
  d = EdgeServer(args.name)
  d.start()
