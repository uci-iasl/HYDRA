import time
from queue import LifoQueue as Queue
from . import data_types
from .blocks import ImageProducer_distr_output as ImageProducer
from .blocks import Logger, ConsumeImageProduceFeatDet, ConsumeFeatProduceAction, ConsumeAction, Advertizer, \
      TegraLogger, NetLoggerMultiInterface, PosReceiver #, Keyboard
from .config import *
import dronekit
from dronekit import VehicleMode
from .SuperDrone import SuperDrone
from .FakeDrone import FakeDrone
from .FlightLog import *
from threading import Thread
import argparse
import numpy as np
import copy
from threading import Lock
from influxdb import InfluxDBClient
from devices.utils import *
from devices.Decisor import Decisor

class Device(Thread):
  def __init__(self, iid, start_time=time.time(), solo_edge=False, adaptive_fr = False, 
          adaptive_pipes=True, is_nano=False, model="", ismac=False, conn_string=None, input_type="zombies",
               db_host=None, verbose=False):
    Thread.__init__(self)
    global VERBOSE
    VERBOSE = verbose
    self.id = iid
    self.start_time = start_time
    self.modules = {}
    self.outfile = open(os.path.join(LOGS_PATH, DEVICE_LOG.format(start_time)), "w+")
    self.queues = {"img": Queue(maxsize=Q_DIM), "feat": Queue(maxsize=Q_DIM), "action": Queue(maxsize=Q_DIM)}
    self.state = {
      "info": Logger(name="timing", outfile = self.outfile, category="flux_point", start=self.start_time, cats=["id", "from/to", "type", "auth"]),
      "dev_id": self.id,
      "lock": Lock(),
      "is_running": True,
      "mode": "explore",
      "start_time": self.start_time,
      "sample_options": {"camera": {"intra_frame": 1/(float(INITIAL_FRAME_RATE)), "prec": 0.1}},
      "active_pipelines": {"": False, self.id: False},
      "pipelines_log": {self.id: []},
      "queues": self.queues,
      "img_out_q": {},
      "adapt_fr": adaptive_fr,
      "adapt_pipes": adaptive_pipes,
      "input": int(input_type) if RepresentsInt(input_type) else input_type,
      "ismac": ismac,
      "model": model if len(model) > 0 else "ssd_mobilenet_v1_coco_2018_01_28",
      }
    if db_host is not None:
      print("--->Adding DataBase!!!")
      self.state["client"] = InfluxDBClient(host=db_host, port=8086, username='admin', password='admin')
    self.modules["advert"] = Advertizer(state=self.state)
    print("added advertizer")
    if is_nano:
       print("starting tegrastats")
       self.modules["tegrastats"] = TegraLogger(state=self.state, freq=10)
       print("starting netstats")
       self.modules["netstats"] = NetLoggerMultiInterface(state=self.state, freq=10)
    self.modules["camera"] = ImageProducer(out_q=self.state["img_out_q"], state=self.state)
    if not solo_edge:
      print("It is NOT solo edge")
      self.state["img_out_q"] = {self.id: self.queues["img"]}
      self.modules["img2feat"] = ConsumeImageProduceFeatDet(in_q=self.queues["img"], 
                        out_q=[self.queues["feat"], ], state=self.state, pipe=self.id)
      self.modules["feat2act"] = ConsumeFeatProduceAction(in_q=self.queues["feat"], 
                        out_q=[self.queues["action"], ], state=self.state, pipe=self.id)
    else:
      print("It is NOT solo edge")
      self.state["active_pipelines"][self.id] = False
    self.drone = None
    if conn_string is not None:
      print("Connecting to {}".format(conn_string))
      self.drone = dronekit.connect(conn_string, baud=57600, wait_ready=True, vehicle_class=SuperDrone)
      flight_log = FlightLog(vehicle=self.drone, start_time=start_time)
      flight_log.start()
    else:
      print("Connecting to FakeDrone")
      self.drone = FakeDrone()
    self.modules["act"] = ConsumeAction(in_q=self.queues["action"], drone=self.drone, state=self.state)
    self.state["pos"] = {self.id: self.drone.location}
    self.modules["pos_receiver"] = PosReceiver(state=self.state)
    self.start_explore = time.time()
    self.last_counted = {}
    self.policy_maker = Decisor(self.state)

  def run(self):
    self.drone.arm_and_takeoff(CLIENT_INITIAL_ALTITUDE)
    count = 0
    for t_name in self.modules:
      print("Starting {}".format(t_name))
      self.modules[t_name].setDaemon(True)
      self.modules[t_name].start()
      print("Started " + t_name)
    print("Device Started")
    self.policy_maker.pipeline_update()
    while self.state["is_running"]:
      self.policy_maker.pipeline_update()
      num_pipes = sum([1 for k in self.state["active_pipelines"] if self.state["active_pipelines"][k]])
      # print(self.state["active_pipelines"])
      if 'client' in self.state:
        posdata = [{"measurement": "pipelines",
                    "tags": {"id": self.state["dev_id"]},
                    "fields": {"active": num_pipes, 
                    "available": len(self.state["active_pipelines"]), },
                    "time": int(time.time() * 1000)}]
        self.state["client"].write_points(posdata, database=DBNAME, time_precision='ms', protocol='json')
      # if "explore" in self.state["mode"]:
      #   if self.start_explore + EXPLORE_INTERVAL < time.time() and self.state["adapt_pipes"]:
      #     self.pipeline_update()
      #   else:
      #     pass
      # else:
      #   if self.state["adapt_pipes"]:
      #     self.pipeline_update()
      time.sleep(PIPELINE_UPDATE_PRECISION)
      count += 1

  def is_running(self,):
    return self.state["is_running"]
  
  def stop(self):
    self.state["info"].save()
    self.outfile.close()
    for m in self.modules:
      m.stop()




# Serial port issues on Ubuntu?
# sudo usermod -a -G dialout $USER
# sudo apt-get remove modemmanager
