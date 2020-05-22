import time
import cv2
from queue import LifoQueue as Queue
from queue import Empty
import threading
from threading import Thread
from inspect import getframeinfo, stack
import socket
import pickle
import _thread
import numpy as np
import subprocess
from statistics import mean
import tensorflow as tf
import sys
from typing import List, Dict
from collections import defaultdict
from .get_os_info import *
from . import data_types
from . import net_p3
from .config import *
import select
import numpy as np
import cv2

if TRT:
  import ctypes
  import pycuda.autoinit
  import pycuda.driver as cuda
  from .blocks_detection_trt import Model
else:
  from .blocks_detection import Model

FINE_LOGS = True
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

# emergency_outfile = open("logs/emergency{}.csv".format(time.time()), "w")

class Module(Thread):
  def __init__(self, in_q, out_q, state, pipe="", name=None):
    Thread.__init__(self)
    self.in_q = in_q
    self.out_q = out_q
    self.pipe = pipe
    self.dev_id = state["dev_id"]
    # self.log = state["monitor"]
    self.info = state["info"]
    self.dev_is_running = state["is_running"]
    self.is_running = True
    self.state = state
    self.pipe = pipe
    self.last_put = {}
    if name is not None:
      self.setName(name)

  def stop(self):
    self.is_running = False


class ImageProducer(Module):
  def __init__(self, out_q, state, pipe=""):
    Module.__init__(self, None, out_q, state, pipe, "image_prod")
    if state["input"] == "zombies":
      self.capture = cv2.imread("data/6zombies_small.jpg")
    else:
      self.capture = cv2.VideoCapture(INPUT_VIDEO)
      self.capture.set(3, 400)
      self.capture.set(4, 300)
      self.last_taken = self.capture.read()[1]
    # self.capture.set(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO, 0)

  def run(self):
    # self.log.add(category="img_prod", message="{}".format(time.time()))
    count = 0
    self.is_running = True
    while self.is_running:  # and self.dev_is_running: # and self.pipe in self.state["active_pipelines"]:
      time_frame_taken = time.time()
      if self.state["input"] == "zombies" or isinstance(self.capture, np.ndarray):
        ret, image = True, self.capture
      else:
        ret, image = self.capture.read()
      if ret:
        # print("Img taken")
        for qpipe in self.state["img_out_q"]:
          # print(qpipe)
          q = self.state["img_out_q"]
          while q.qsize() >= MAX_IMG_QUEUE:
            q.get()
            q.task_done()
        st = time.time()
        # image = cv2.resize(image, (400, 300))
        if JPEG:
          result, image = cv2.imencode('.jpg', image, encode_param)
        img_taken = data_types.Image(image=image, t_frame=time.time(), frame_auth=self.dev_id)
        # self.info.add(category="img_prod", fields={"id": img_taken.t_frame, "from/to": "img_prod", "type": "img", "auth": self.dev_id})
        # print("blocks 3.2 line 69")
        if FINE_LOGS:
          self.info.save_action(img_taken)
        if VERBOSE:
          print(self.out_q)
        for q in self.out_q:
          if q not in self.last_put:
            self.last_put[q] = -1
          item = None
          to_put = img_taken
          try:
            item = q.get(False)
            if item:
              q.task_done()
              if to_put.t_frame < item.t_frame:
                to_put = item
          except Empty as e:
            pass
          except Exception as e:
            raise e
          finally:
            if to_put.t_frame > self.last_put[q]:
              q.put(to_put)
              self.last_put[q] = to_put.t_frame
              # print("Img producer {}, {}".format(q, to_put.t_frame))
            else:
              print("IMG producer NO PUT!!!", time.time(), self.last_put, to_put.t_frame)
      # print("t = {0:.2f} : Img taken".format(time.time()-self.state["start_time"]))
      time_before_next_frame = self.state["sample_options"]["camera"]["intra_frame"] + time_frame_taken - time.time()
      time.sleep(max(time_before_next_frame, 0))
    print("Img called stop")
    self.stop()


class ImageProducer_distr_output(Module):
  def __init__(self, out_q, state, pipe=""):
    Module.__init__(self, None, out_q, state, pipe, "ImgProdDistr")
    self.capture = cv2.VideoCapture(state["input"])
    self.capture.set(3, 480)
    self.capture.set(4, 360)
    self.last_taken = cv2.resize(self.capture.read()[1], (300, 300))
    self.next_q = -1

  def run(self):
    count = 0
    self.is_running = True
    while self.is_running:  # and self.dev_is_running: # and self.pipe in self.state["active_pipelines"]:
      if len(self.state["img_out_q"]) < 1:
        if VERBOSE:
          print("NO pipes, no party! - Not producing images due to no pipes active")
        time.sleep(self.state["sample_options"]["camera"]["intra_frame"])
      img_out_q_keys = list(self.state["img_out_q"])
      for q_pipe in img_out_q_keys:
        start = time.time()
        if VERBOSE:
          print("Running {} of {}".format(q_pipe, list(self.state["img_out_q"])))
        q = self.state["img_out_q"][q_pipe]
        time_frame_taken = time.time()
        #print("153 - {}".format(time_frame_taken-start))
        if VERBOSE:
          print("GETTING IMG")
        if isinstance(self.capture, np.ndarray):
          ret, image = True, self.capture
        else:
          for _ in range(1):
            ret, image = self.capture.read()
        st1 = time.time()
        # print("161 - {}".format(st1 - time_frame_taken))
        # image = cv2.resize(image, (300, 300))
        # print("At image capture distr image has shape {}".format(image.shape))
        if ret:
          while q.qsize() >= MAX_IMG_QUEUE:
            q.get()
            q.task_done()
          st = time.time()
          #print("169 - {}".format(time.time() - st1))
          #print("Up to st it takes {}".format(start - st))
          if JPEG:
            result, image = cv2.imencode('.jpg', image, encode_param)
          img_taken = data_types.Image(image=image, t_frame=time.time(), frame_auth=self.dev_id)
          # self.info.add(category="img_prod", fields={"id": img_taken.t_frame, "from/to": "img_prod", "type": "img", "auth": self.dev_id})
          if FINE_LOGS:
            self.info.save_action(img_taken)
          if VERBOSE:
            print("self.out.q from IMAGE Producer Distr {}".format(str(self.out_q)))
          if q not in self.last_put:
            self.last_put[q] = -1
          item = None
          to_put = img_taken
          try:
            item = q.get(False)
            if item:
              q.task_done()
              if to_put.t_frame < item.t_frame:
                to_put = item
          except Empty as e:
              #print(e, "EMPTY AT 191")
              pass
          except Exception as e:
            raise e
          finally:
            #print("Takes {} from st to here".format(time.time() - st))
            if to_put.t_frame > self.last_put[q]:
              q.put(to_put)
              self.last_put[q] = to_put.t_frame
              #print("Img producer {}, {}".format(q_pipe, to_put.t_frame))
            else:
              print("IMG producer NO PUT!!!", time.time(), self.last_put, to_put.t_frame)
        time_before_next_frame = (self.state["sample_options"]["camera"]["intra_frame"] / len(
          self.state["img_out_q"]) + time_frame_taken - time.time())
        time.sleep(max(time_before_next_frame, 0))
        # print("Intra frame time: {:0.3f}, time left: {:0.3f}".format(self.state["sample_options"]["camera"]["intra_frame"] / len(self.state["img_out_q"]), time_before_next_frame))

    print("Img called stop")
    self.stop()


class ConsumeImageProduceFeatDet(Module):
  def __init__(self, in_q, out_q, state, pipe, save_img=False, condition=None):
    Module.__init__(self, in_q, out_q, state, pipe, "img2feat")
    self.setName("Data Analysis")
    print("Loading model {}".format(self.state["model"]))
    # self.detector.predict(self.detector.load_image_into_numpy_array_cv('data/6zombies_small.jpg'))
    self.artificial_delay = 0. if "UAV" in self.dev_id else 0
    self.save_img = save_img

  def run(self):
    if TRT:
      self.cuda_ctx = cuda.Device(0).make_context()
    self.detector = Model(self.state["model"])
    while self.is_running and self.dev_is_running:
      try:
        image = self.in_q.get(timeout=Q_READ_TIMEOUT)
        if VERBOSE:
          print(time.time(), "PICK UP DDN", image.t_frame)
        while self.in_q.qsize() >= MAX_IMG_QUEUE:
          tmp = self.in_q.get()
          if VERBOSE:
            print(time.time(), "PICK UP DDN", image.t_frame)
          if (image.t_frame < tmp.t_frame):
            image = tmp
          self.in_q.task_done()
      except Empty as e:
        if VERBOSE:
          (self.getName() + " - Timeout expired")
          print(e)
        self.is_running = False
      else:
        self.in_q.task_done()
        if self.state["active_pipelines"][self.pipe] or not self.state["adapt_pipes"]:
          image_exp = image
          st = time.time()
          if JPEG:
            image_exp = cv2.imdecode(image.get_value(), 1)
          if not TRT:
            img2feed = np.expand_dims(image_exp, axis=0)
          else:
            img2feed = image_exp
          bbox = self.detector.predict(img2feed, 'person')
          # print(bbox)
          if bbox:
            size = image_exp.shape
            #print("Image shape at ConsumeImage = {}",format(size))
            if TRT:
              x1, y1, x2, y2 = bbox[0][0]
            else:
              y1, x1, y2, x2 = bbox[0][0]
            #print("Box {}".format(bbox[0]))
            x1 = int(float(x1) * float(size[1]))
            x2 = int(float(x2) * float(size[1]))
            y1 = int(float(y1) * float(size[0]))
            y2 = int(float(y2) * float(size[0]))
            cv2.rectangle(image_exp, (x1, y1), (x2, y2), (0, 0, 255), 3)
          time.sleep(self.artificial_delay)
          if 'client' in self.state:
            posdata = [{"measurement": "pred_delay",
                        "tags": {"id": self.state["dev_id"]},
                        "fields": {"delay": time.time() - st},
                        "time": int(time.time() * 1000)}]
            self.state["client"].write_points(posdata, database=DBNAME, time_precision='ms', protocol='json')
          if bbox:
            x1, y1, x2, y2 = bbox[0][0]  # first is example, second is bbox
            pos_img = ((x1 + x2) / 2), ((y1 + y2) / 2), ((x2 - x1) * (y2 - y1))
          else:
            pos_img = CENTER
          feat_inst = data_types.Detection(x=pos_img[0], y=pos_img[1], area=pos_img[2], t_frame=image.get_time(),
                                           frame_auth=image.get_auth(), t_det=time.time(), det_auth=self.dev_id)
          cv2.putText(
            image_exp,
            "{} {} {}".format(pos_img[0], pos_img[1], pos_img[2]),  # text
            (10,50),  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            1,  # font size
            (209, 80, 0, 255),  # font color
            3)  # font stroke
          if VERBOSE:
            print(feat_inst)
        for q in self.out_q:
          if q not in self.last_put:
            self.last_put[q] = -1
          item = None
          to_put = feat_inst
          try:
            item = q.get(False)
            if item:
              q.task_done()
              if to_put.t_frame < item.t_frame:
                to_put = item
          except:
            pass
          finally:
            if to_put.t_frame > self.last_put[q]:
              if VERBOSE:
                print(time.time(), "PUT DOWN DNN", to_put.t_frame)
              q.put(to_put)
              self.last_put[q] = to_put.t_frame
              if VERBOSE:
                print("Feat production PUT")
            else:
              if VERBOSE:
                print(time.time(), self.last_put, to_put.t_frame)
          global COUNT
          cv2.imwrite("data/img_{}.jpg".format(COUNT), image_exp)
          COUNT += 1

    def stop(self):
      del self.detector
      self.cuda_ctx.pop()
      del self.cuda_ctx


class ConsumeFeatProduceAction(Module):
  def __init__(self, in_q, out_q, state, pipe):
    Module.__init__(self, in_q, out_q, state, pipe, "feat2act")

  def run(self):
    while self.dev_is_running:
      try:
        if VERBOSE:
          print("GET in Prod action")
        feat = self.in_q.get(timeout=Q_READ_TIMEOUT)

        if VERBOSE:
          print(time.time(), "PICK UP FEAT", feat.t_frame)
        # self.info.save_action(feat)
      except Empty as e:
        print(self.getName() + " - Timeout expired")
        print(e)
        self.stop()
      else:
        self.in_q.task_done()
        action = self.detection2movement(feat)
        #print("Feat {}".format(feat))
        #print("action {}".format(str(action)))
        if VERBOSE:
          print(action)
          print("action produced")
        if self.is_running:
          if VERBOSE:
            print(self.out_q)
          for q in self.out_q:
            if q not in self.last_put:
              self.last_put[q] = -1
            item = None
            to_put = action
            try:
              item = q.get(False)
              if item:
                if VERBOSE:
                  print("DISCARD ACT")
                q.task_done()
                if to_put.t_frame < item.t_frame:
                  to_put = item
            except:
              if VERBOSE:
                print(q)
            finally:
              if VERBOSE:
                print("in finally line 204")
              if to_put.t_frame > self.last_put[q]:
                if VERBOSE:
                  print(time.time(), "PUT DOWN ACT", feat.t_frame)
                q.put(to_put)
                self.last_put[q] = to_put.t_frame
                if VERBOSE:
                  print("Produce action PUT")
              else:
                if VERBOSE:
                  print("DISCARD ACT")
                  print(time.time(), self.last_put, to_put.t_frame)

  ## Note: here we convert from 2D picture, to the 3D world.
  ## In this case, we pass FB to x, LR to y, UD to z                  
  # ",".join([str(pos) for pos in [feat.x, feat.y]]) + '\n')
  def detection2movement(self, detection):
    tmp = np.array((detection.x, detection.y, detection.area))
    d = CENTER - tmp
    lr_action, ud_action, fb_action = 0, 0, 0
    if abs(d[0]) > X_CUTOFF:  # does it make sense to move?
      if d[0] > 0:  # target on my left?
        lr_action = LEFT
      else:
        lr_action = RIGHT
    else:
      pass  # not worth of it
    if abs(d[1]) > Y_CUTOFF:
      if d[1] > 0:  # target up?
        ud_action = UP
      else:
        ud_action = DOWN
    else:
      pass  # not worth of it
    if abs(d[2]) > AREA_CUTOFF*TARGET_AREA:
      if d[2] > 0:
        fb_action = BACKWARD
      else:
        fb_action = FORWARD
    else:
      pass  # not worth moving
    
    ##DISABLE Z AXIS!!!!#########
    #ud_action = 0
    #fb_action = 0
    ##DISABLE Z AXIS!!!!#########
    #print("-->Produced action is (fb, lr, ud): {}".format(str((fb_action, lr_action, ud_action))))
    action = data_types.Action(x_act=fb_action, y_act=lr_action, z_act=ud_action, t_frame=detection.t_frame, 
                               frame_auth=detection.frame_auth, t_det=detection.t_det, 
                               det_auth=detection.det_auth, t_act=time.time(), act_auth=self.dev_id)
    return action


class ConsumeAction(Module):
  def __init__(self, in_q, state, drone, pipe=""):
    Module.__init__(self, in_q, [], state, pipe, "consAct")
    self.setName("Operator")
    self.last_act = None
    self.drone = drone
    self.mov_dur = MOVEMENT_DURATION

  def run(self):
    fields_dict = {"id": None, "from/to": "consume_action_{}".format(self.pipe)}
    while self.is_running and self.dev_is_running:
      action_list = []
      action2take = None
      try:
        action_list.append(self.in_q.get(timeout=Q_READ_TIMEOUT))
        self.in_q.task_done()
        while not self.in_q.empty():
          action_list.append(self.in_q.get(timeout=Q_READ_TIMEOUT))
          self.in_q.task_done()
      except Empty as e:
        if VERBOSE:
          print(self.getName() + " - Timeout expired")
          print(e)
          print("consume action stopping")
        self.stop()
      else:
        action2take = self.choose_action(action_list, self.last_act)
        if action2take is not None:
          time_of_action = action2take.t_act #this could be time.time()
        else:
          time_of_action = time.time()
        for e in action_list:
          self.state["pipelines_log"][e.det_auth].append((e.t_frame, time_of_action - e.t_frame))
          if 'client' in self.state:
            posdata = [{"measurement": "tot_delay",
                        "tags": {"id": e.det_auth},
                        "fields": {"value": time_of_action-e.t_frame},
                        "time": int(time.time() * 1000)}]
            self.state["client"].write_points(posdata, database=DBNAME, time_precision='ms', protocol='json')
          if FINE_LOGS:
            self.info.write(category='0', fields={**fields_dict, **{"id": e.t_frame, 'time': time.time()}})
          if e is action2take:
            self.info.save_action(e, True, time_of_action)
          else:
            self.info.save_action(e, False, time_of_action)

      if action2take is not None:
        if 'client' in self.state:
          posdata = [{"measurement": "action",
                      "tags": {"id": action2take.det_auth},
                      "fields": {"fb":action2take.x_act, "lr":action2take.y_act, "ud":action2take.z_act},
                      "time": int(time.time() * 1000)}]
          self.state["client"].write_points(posdata, database=DBNAME, time_precision='ms', protocol='json')
        print(".", end="")
        # print("action2take {}".format(str((action2take.x_act, action2take.y_act, action2take.z_act))))
        sys.stdout.flush()
        action2take.t_act = time.time()
        # info2print = [str(e) for e in
        #               [action2take.det_auth, action2take.t_frame, action2take.t_act - action2take.t_frame]]
        # info2print.append(str("-")) if self.last_act is None else info2print.append(
        #   str(action2take.t_act - self.last_act.t_act))
        since_last_action = None
        if action2take is not None and self.last_act is not None:
          since_last_action = action2take.t_act-self.last_act.t_act
        self.last_act = action2take
        if self.state["adapt_fr"]:
          self.state["sample_options"]["camera"]["intra_frame"] = ((time.time() - action2take.t_frame) * ALPHA + (
                    1 - ALPHA) * self.state["sample_options"]["camera"]["intra_frame"])
          self.mov_dur = self.mov_dur
        if 'client' in self.state and self.state["client"] is not None:
          self.state['client'].write(["{} value={}".format('frame_rate', len(self.state["img_out_q"])/self.state["sample_options"]["camera"]["intra_frame"])],
                                          {'db':DBNAME},204,'line')
        if "client" in self.state and self.state["client"] is not None and since_last_action is not None:
          self.state['client'].write(["{} value={}".format('since_last_action', since_last_action)],
                                     {'db': DBNAME}, 204, 'line')
        if VERBOSE:
          print("MOVE {} {} {}".format(action2take.det_auth, action2take.t_act - action2take.t_frame, str((action2take.x_act, action2take.y_act, action2take.z_act))))
        self.drone.move(action2take.x_act, action2take.y_act, action2take.z_act,
                        duration=max(self.state["sample_options"]["camera"]["intra_frame"] / len(self.state["img_out_q"]), MIN_MOVING_TIME))  # self.state["sample_options"]["camera"]["intra_frame"]/2

  def choose_action(self, list_of_actions, last_action_taken):
    ret = None
    s = "List of actions: \n"
    for e in list_of_actions:
      if VERBOSE:
        print("ACTION: {} {}  ---only lab---  {}".format(e.det_auth, e.t_act - e.t_frame, e.t_det - e.t_frame))
      s += str(e) + '\n'
      if ret is not None:
        if ret.t_frame < e.t_frame:
          ret = e
      else:
        ret = e
    if last_action_taken is not None:
      if not (int(last_action_taken.t_frame * 100) < int(ret.t_frame * 100)):
        return None
    # s += "The chosen one\n" + str(ret)
    # print(s)
    # self.info.add(category='action', message="{}".format(s))
    return ret


class Logger:
  def __init__(self, name, outfile=None, category=None, start=time.time(), cats=["message", ], s=None):
    self.name = name
    self.info = []
    self.start_time = start
    if outfile is None:
      outfile = open(os.path.join(LOGS_PATH, "logger_outfile_NONE_{}.csv".format(start)), "w")
    self.outfile = outfile
    if category is None:
      category = "category"
    self.header = ["time", "lineno", category] + cats + ["taken, capt-to-action"]
    # print("self.header", self.name, self.header)
    self.cats = cats
    self.state = s
    self.file_lock = threading.Lock()

  def write(self, category, fields):
    self.file_lock.acquire()
    time_cat_fields = [category, ] + [fields[e] for e in fields]
    line = ','.join([str(e) for e in time_cat_fields]) + '\n'
    self.outfile.write(line)
    # self.outfile.flush()
    self.file_lock.release()

  def write_add_time(self, category, fields):
    self.file_lock.acquire()
    time_cat_fields = [time.time(), category] + [fields[e] for e in fields]
    line = ','.join([str(e) for e in time_cat_fields]) + '\n'
    self.outfile.write(line)
    # self.outfile.flush()
    self.file_lock.release()

  def add(self, category, fields):
    caller = getframeinfo(stack()[1][0])
    # print("fields", fields)
    self.info.append([time.time() - self.start_time, caller.lineno, category] + [fields[e] for e in self.cats])

  def save_action(self, act, taken=False, time_taken=None):
    if VERBOSE:
      print("Writing action in file")
    timing = time.time()
    extras = ", {}, {}".format(1 if taken else 0,
                               timing - act.t_frame if time_taken is None else time_taken - act.t_frame)
    # print(act.nice_str() + extras)
    self.file_lock.acquire()
    self.outfile.write(act.nice_str() + extras + "\n")
    # self.outfile.flush()
    self.file_lock.release()

  def __str__(self):
    s = ",".join(self.header) + "\n"
    for lst in self.info:
      s += ",".join([str(e) for e in lst]) + "\n"
    return s

  def save(self, filename=None):
    if filename is None:
      filename = os.path.join(LOGS_PATH, "logger_old_{}_{:0.2f}.csv".format(self.name, self.start_time))
    with open(filename, "w") as f:
      f.write(self.__str__())
      f.close()


class Advertizer(Thread):
  def __init__(self, state):
    Thread.__init__(self)
    self.state = state
    self.ips = (subprocess.check_output(['hostname', '--all-ip-addresses'])).decode().strip().split()
    self.discovery_socks = []
    for e in self.ips:
      tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      tmp.bind((e, 0))
      tmp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
      self.discovery_socks.append(tmp)
    if True:
      print("Initiated sockets on {}".format([e.getsockname() for e in self.discovery_socks]))

  def run(self):
    print("Advertizer in Run")
    while True:
      self.update_ips() # All the IPs that this device has (i.e. one for each net interface)
      for sock in self.discovery_socks:
        try:
          sock.sendto(pickle.dumps(MESSAGE.format(self.state["dev_id"])), ("<broadcast>", UDP_DISCOVERY_PORT))
          sock.settimeout(2)
          if VERBOSE:
            print("Socket on {} ".format(sock.getsockname()[0]))
          #print("Waiting for data")
          data, addr = sock.recvfrom(1024)
          #print("received something")
          rcvd_obj = pickle.loads(data)
          if True:
            print("client received - {}".format(rcvd_obj))
          # print("rcvd_obj[name]self.state[active_pipelines]", rcvd_obj["name"], self.state["active_pipelines"])
          if rcvd_obj["name"] not in self.state["active_pipelines"]:
            self.add_edge_pipeline(rcvd_obj["name"])
            c = Connector(ip=rcvd_obj["ip"], port=rcvd_obj["port"], pipe=rcvd_obj["name"], state=self.state)
            c.start()
        except socket.timeout as e:
          if VERBOSE:
            print("Socket on {} timed out".format(sock.getsockname()[0]))

  def stop(self):
    for sock in self.discovery_socks:
      sock.close()

  def update_ips(self):
    # Keep in self.ips the ones for which we already have a socket open
    # Open new sockets for the new ones
    current_ips = (subprocess.check_output(['hostname', '--all-ip-addresses'])).decode().strip().split()
    self.discovery_socks = []
    new_ips = [ip for ip in current_ips if ip not in self.ips]
    self.ips += new_ips
    for e in self.ips:
      tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      tmp.bind((e, 0))
      tmp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
      self.discovery_socks.append(tmp)

  def add_edge_pipeline(self, pipe):
    self.state['lock'].acquire()
    self.state["active_pipelines"][pipe] = True
    self.state["pipelines_log"][pipe] = []
    self.state["queues"]["{}_in".format(pipe)] = Queue(maxsize=Q_DIM)
    self.state["queues"]["{}_out".format(pipe)] = Queue(maxsize=Q_DIM)
    self.state["img_out_q"][pipe] = self.state["queues"]["{}_in".format(pipe)]
    self.state["lock"].release()
    print("-> Added this q {}".format("{}_out".format(pipe)))
    print("Available queues are : {}".format(self.state["img_out_q"]))


class Connector(Module):
  def __init__(self, ip, port, state, pipe, in_q=None, out_q=None):
    Module.__init__(self, in_q=state["queues"]["{}_in".format(pipe)], out_q=state["queues"]["action"], state=state,
                    pipe=pipe, name="Connector")
    # Create a TCP/IP socket
    self.ip = ip
    # if pipe in "active_pipelines":
    # raise ValueError("{} already an active pipeline".format((name, self)))
    # self.state["active_pipelines"].append((pipe, None, None))
    print("In CONNECTOR")
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket created")
    self.buff_size = BUFF_SIZE
    self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    # Connect the socket to the port where the server is listening
    server_address = (ip, port)
    print('CLIENT ---> connecting to {} port {}'.format(*server_address))
    time.sleep(0.1)
    print("Connecting with {}".format(server_address))
    self.sock.connect(server_address)
    self.pipe = pipe
    print("After connection")
    _thread.start_new_thread(self.ready_to_reply, ())

  def run(self):
    rem = b""
    fields_dict = {"id": None, "from/to": "connector_{}".format(self.pipe)}
    while self.pipe in self.state["active_pipelines"]:
      try:
        if VERBOSE:
          print("Waiting to receive a message")
        t1, t2, data, rem = net_p3.receive_fat_msg(self.sock, rem)
        # print("Received stuff - putting it in the q")
        if VERBOSE:
          print(time.time(), "Message received")
        if self.state["active_pipelines"][self.pipe] or not self.state["adapt_pipes"]:
          if self.state["queues"]["action"] not in self.last_put:
            self.last_put[self.state["queues"]["action"]] = -1
          tmp = pickle.loads(data)
          if FINE_LOGS:
            self.info.write(category="rcvd_data", fields={**fields_dict, **{"id": tmp.t_frame, "time": time.time()}})
          if VERBOSE:
            print(time.time(), "Message received", tmp.t_frame, self.sock.getsockname())
          item = None
          to_put = tmp
          try:
            item = self.state["queues"]["action"].get(False)
            if item:
              self.state["queues"]["action"].task_done()
              if to_put.t_frame < item.t_frame:
                to_put = item
            if item:
              self.state["queues"]["action"].task_done()
          except:
            # print("_"*80)
            pass
          finally:
            if VERBOSE and False:
              print("Connector PUT")
            if to_put.t_frame > self.last_put[self.state["queues"]["action"]]:
              to_put.t_act = time.time()
              self.state["queues"]["action"].put(to_put)
              self.last_put[self.state["queues"]["action"]] = to_put.t_frame
            else:
              if VERBOSE:
                print(time.time(), self.last_put, to_put.t_frame)
            if FINE_LOGS:
              self.info.write(category="put_out_q",
                              fields={**fields_dict, **{"id": to_put.t_frame, "time": time.time()}})
        else:
          print("Got packet but {} is not active".format(self.pipe))
      except ConnectionResetError as conn_reset:
        if self.pipe in self.state["active_pipelines"]:
          del self.state["active_pipelines"][self.pipe]
        if self.pipe in self.state["pipelines_log"]:
          del self.state["pipelines_log"][self.pipe]
      except ValueError:
        if self.pipe in self.state["active_pipelines"]:
          del self.state["active_pipelines"][self.pipe]
        if self.pipe in self.state["pipelines_log"]:
          del self.state["pipelines_log"][self.pipe]
      except Exception as e:
        if VERBOSE:
          print("_" * 80)
        if self.pipe in self.state["active_pipelines"]:
          del self.state["active_pipelines"][self.pipe]
        if self.pipe in self.state["pipelines_log"]:
          del self.state["pipelines_log"][self.pipe]

  def add_edge_pipeline(self, pipe):
    self.state['lock'].acquire()
    self.state["active_pipelines"][pipe] = True
    self.state["pipelines_log"][pipe] = []
    self.state["queues"]["{}_in".format(pipe)] = Queue(maxsize=Q_DIM)
    self.state["queues"]["{}_out".format(pipe)] = Queue(maxsize=Q_DIM)
    self.state["img_out_q"][pipe] = self.state["queues"]["{}_in".format(pipe)]
    self.state["lock"].release()
    # print("-> Added this q {}".format("{}_out".format(pipe)))
    print("Available queues are : {}".format(self.state["img_out_q"]))

  def remove_edge_pipeline(self, pipe):
    self.state['lock'].acquire()
    if self.pipe in self.state["active_pipelines"]:
      del self.state["active_pipelines"][self.pipe]
    if self.pipe in self.state["pipelines_log"]:
      del self.state["pipelines_log"][self.pipe]
    if pipe in self.state["img_out_q"]:
      del self.state["img_out_q"][pipe]
    if "{}_in".format(pipe) in self.state["queues"]:
      del self.state["queues"]["{}_in".format(pipe)]
    self.state["lock"].release()
    print("-> Added this q {}".format("{}_out".format(pipe)))
    print("Available queues are : {}".format(self.state["img_out_q"]))

  def ready_to_reply(self):
    fields_dict = {"id": None, "from/to": "replying_{}".format(self.pipe)}
    while self.is_running:
      try:
        self.sock.setblocking(1)
        if VERBOSE:
          print("Waiting image in my queue at ready_to_reply {}".format(self.ip))
        image = self.in_q.get(timeout=Q_READ_TIMEOUT)
        if VERBOSE:
          print("Got at ready_to_reply")
          print(time.time(), "Got Image at ready_to_reply", image.t_frame, [str(e) for e in self.sock.getsockname()])
        # print(self.sock.getsockname())
        while self.in_q.qsize() > MAX_IMG_QUEUE:
          tmp = self.in_q.get()
          if (image.t_frame < tmp.t_frame):
            if FINE_LOGS:
              self.info.write(category="trash", fields={**fields_dict, **{"id": image.t_frame, "time": time.time()}})
            image = tmp
          else:
            if FINE_LOGS:
              self.info.write(category="trash", fields={**fields_dict, **{"id": image.t_frame, "time": time.time()}})
          self.in_q.task_done()
        if FINE_LOGS:
          self.info.write(category="pop_from_q", fields={**fields_dict, **{"id": image.t_frame, "time": time.time()}})
      except Empty as e:
        if VERBOSE:
          print(self.getName() + " - Timeout expired")
          print(e)
      else:
        self.in_q.task_done()
        if self.pipe not in self.state["active_pipelines"]:
          break
        if self.state["active_pipelines"][self.pipe] or not self.state["adapt_pipes"]:
          # print("sending stuff over")
          if FINE_LOGS:
            self.info.write(category="tx_start", fields={**fields_dict, **{"id": image.t_frame, "time": time.time()}})
          st = time.time()
          encoded_img = pickle.dumps(image)
          try:
            ret_from_fatsend = False
            while not ret_from_fatsend:
              ret_from_fatsend = net_p3.send_fat_msg(self.sock, encoded_img, 0., 0.)
              #print("ret in the loop, ", ret_from_fatsend)
              time.sleep(0.025)
            #print("Ret = ", ret_from_fatsend)
          except BrokenPipeError:
            self.remove_edge_pipeline(self.pipe)
          if 'client' in self.state:
            posdata = [{"measurement": "send_msg",
                        "tags": {"id": self.pipe},
                        "fields": {"delay": time.time() - st, "size": len(encoded_img)},
                        "time": int(time.time() * 1000)}]
            self.state["client"].write_points(posdata, database=DBNAME, time_precision='ms', protocol='json')
          if FINE_LOGS:
            self.info.write(category="tx_stop", fields={**fields_dict, **{"id": image.t_frame, "time": time.time()}})

"""
TODO: 1. extend to different format
      2. recognize format automatically
"""
class TegraLogger(Thread):
  def __init__(self, state, log_file_path=None, freq=10):
    """
    @:arg - log file name
    @:arg - sampling frequency
    """
    Thread.__init__(self)
    cmds = ["tegrastats", "--interval", str(int(1000 / freq))]
    if log_file_path is None:
      log_file_path = "logs/{}_parsed_tegrastats.csv".format(int(state["start_time"]))
    self.p = subprocess.Popen(cmds, stdout=subprocess.PIPE)
    self.log_file = open(log_file_path, 'w+')
    self.state = state

  def run(self):
    stats, others = {}, {}
    text = None
    while self.state["is_running"]:
      current_stat = self.p.stdout.readline().decode().strip()
      if current_stat == '':
        time.sleep(0.01)
        continue
        #  raise ValueError("Tegrastats error detected")
      fields = current_stat.split(" ")
      stats["ram_used"], stats["ram_tot"] = [int(e) for e in fields[1][:-2].split("/")]
      others["cpu_perc_freq"] = [[int(e.split("%@")[0]), int(e.split("%@")[1])] for e in fields[5][1:-1].split(",")]
      stats["avg_used"] = mean([perc / 100. for perc, freq in others["cpu_perc_freq"]])
      stats["avg_freq"] = mean([freq for perc, freq in others["cpu_perc_freq"]])
      # Weighted average frequency
      stats["w_avg_freq"] = mean([perc / 100. * freq for perc, freq in others["cpu_perc_freq"]])
      # External Memory Control Frequency percentage used
      stats["emc"] = int(fields[7][:-1])
      stats["gpu_used"] = int(fields[9][:-1])
      for i in [16, 18, 20]:
        stats["pom_5v_{}".format(fields[i].split("_")[-1])] = int(fields[i + 1].split("/")[0])
      if text is None:
        text = str("time") + "," + ",".join([str(e) for e in sorted(stats)])
        self.log_file.write(text + '\n')

      text = str(time.time()) + "," + ",".join([str(stats[e]) for e in sorted(stats)])
      self.log_file.write(text + '\n')
      self.log_file.flush()

  def stop(self):
    self.log_file.close()


class NetLoggerMultiInterface(Thread):
  def __init__(self, state=None, log_file_path=None, freq=10):
    """
    @:arg - dict containing state of the device (uses start_time and is_running)
    @:arg - log file name
    @:arg - sampling frequency
    """
    Thread.__init__(self)
    self.files = {"tcp": None, "signal": None}
    self.fields = self.get_dict()[0]
    for info_type in self.files:
      name = log_file_path = "logs/{}_{}_parsed_netstats.csv".format(int(state["start_time"]), info_type)
      self.files[info_type] = open(log_file_path, 'w+')
      tmp = str("time") + "," + ",".join(self.fields[info_type])
      self.files[info_type].write(tmp + '\n')
    self.state = state
    self.interval = float(1. / freq)  # interval in milliseconds

  def run(self):
    stats, others = {}, {}
    while self.state["is_running"] and not any([self.files[iface].closed for iface in self.files]):
      start = time.time()
      stats = self.get_dict(skip_ssh=True, header=self.fields)
      for info_type in self.files:
        tmp = "\n".join([",".join([str(e) for e in [time.time()] + l]) for l in stats[info_type]])
        self.files[info_type].write(tmp + '\n')
        self.files[info_type].flush()
      tmp = start - time.time() + self.interval
      if tmp > 0:
        time.sleep(tmp + self.interval)

  def stop(self):
    for iface in self.files:
      self.files[iface].close()

  def get_dict(self, skip_ssh=False, header=None):
    wireless = parse_wireless()
    ifconfig = ifconfig_all()
    ip2iface = {}
    for e in ifconfig:
      if "ip" in ifconfig[e]:
        ip2iface[ifconfig[e]["ip"]] = e
    for e in wireless:
      ifconfig[e].update(wireless[e])
    ifconfig_list = [ifconfig[e] for e in ifconfig]

    tcp_info = ss_info_tcp(skip_ssh=skip_ssh)
    ret = {"signal": ifconfig_list, "tcp": tcp_info}
    fields, content = {}, {}
    for info_type in ['signal', 'tcp']:
      if header is None:
        fields[info_type] = sorted(ret[info_type][0])
      else:
        fields = header
      content[info_type] = [[ret[info_type][i][k] if k in ret[info_type][i] else "None" for k in fields[info_type]] for
                            i in range(len(ret[info_type]))]
    if header is None:
      return (fields, content)
    else:
      return content


class PosSender(Thread):
  def __init__(self, state):
    Thread.__init__(self)
    self.state = state
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    if VERBOSE:
      print("Initiated sockets on {}".format(self.sock.getsockname()))

  def run(self):
    while True:
      current_ips = (subprocess.check_output(['hostname', '--all-ip-addresses'])).decode().strip().split()
      for ip in current_ips:
        self.sock.sendto(pickle.dumps({"id": self.state["dev_id"], "pos": self.state["drone"].location.global_frame}),(ip, POS_PORT))
        if VERBOSE:
          print("-PosSender sending current position {} to {}".format(self.state["drone"].location, (ip, POS_PORT)))
      time.sleep(POS_INTERVAL)

class PosReceiver(Thread):
  """
  On the drone
  """
  def __init__(self, state):
    Thread.__init__(self)
    self.state = state
    self.ips = (subprocess.check_output(['hostname', '--all-ip-addresses'])).decode().strip().split()
    self.discovery_socks = []
    self.log_pos = open("logs/posfile_{}.csv".format(self.state["start_time"]), "w")
    for e in self.ips:
      tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      tmp.bind((e, POS_PORT))
      # tmp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
      tmp.setblocking(0)
      self.discovery_socks.append(tmp)
    if VERBOSE:
      print("Initiated sockets on {}".format([e.getsockname() for e in self.discovery_socks]))

  def run(self):
    while True:
      ready_to_read, _, _ = select.select(self.discovery_socks, [], [])
      for e in ready_to_read:
        rcvd_obj = e.recvfrom(BUFF_SIZE)
        self.log_pos.write(",".join([rcvd_obj[0]["id"], str(rcvd_obj[0]["pos"])]) + "\n")
        self.update_dev_pos(pickle.loads(rcvd_obj[0]))

  def update_ips(self):
    # Keep in self.ips the ones for which we already have a socket open
    # Open new sockets for the new ones
    current_ips = (subprocess.check_output(['hostname', '--all-ip-addresses'])).decode().strip().split()
    self.discovery_socks = []
    new_ips = [ip for ip in current_ips if ip not in self.ips]
    self.ips += new_ips
    for e in self.ips:
      tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      tmp.bind((e, 0))
      tmp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
      self.discovery_socks.append(tmp)

  def update_dev_pos(self, rcvd_obj):
    self.state["lock"].acquire()
    self.state["pos"][rcvd_obj["id"]] = rcvd_obj["pos"]
    self.state["lock"].release()
    if VERBOSE:
      print("Updated location is {}".format(str(self.state["pos"])))


if __name__ == "__main__":
  n = NetLoggerMultiInterface({"start_time": time.time(), "is_running": True}, freq=15)
  n.start()
  time.sleep(5)
  n.stop()
