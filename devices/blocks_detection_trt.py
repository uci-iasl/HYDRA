import ctypes

import numpy as np
import cv2
import tensorflow as tf
import tensorrt as trt
import pycuda.driver as cuda



NUM_CLASSES = 90
class Model(object):
  """TrtSSD class encapsulates things needed to run TRT SSD."""
  def __init__(self, model='ssd_mobilenet_v2_coco', input_shape=(300,300), output_layout=7):
    """Initialize TensorRT plugins, engine and conetxt."""
    self.model = model
    self.input_shape = input_shape
    self.output_layout = output_layout
    self.trt_logger = trt.Logger(trt.Logger.INFO)
    self._load_plugins()
    self.engine = self._load_engine()
    self.host_inputs = []
    self.cuda_inputs = []
    self.host_outputs = []
    self.cuda_outputs = []
    self.bindings = []
    self.context = self._create_context()
    self.stream = cuda.Stream()
    self.labels = self.load_labels()

  def _load_plugins(self):
    if trt.__version__[0] < '7':
      ctypes.CDLL("ssd/libflattenconcat.so")
    trt.init_libnvinfer_plugins(self.trt_logger, '')

  def _load_engine(self):
    TRTbin = 'models/ssd/TRT_%s.bin' % self.model
    with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
      return runtime.deserialize_cuda_engine(f.read())

  def _create_context(self):
    for binding in self.engine:
      size = trt.volume(self.engine.get_binding_shape(binding)) * \
           self.engine.max_batch_size
      host_mem = cuda.pagelocked_empty(size, np.float32)
      cuda_mem = cuda.mem_alloc(host_mem.nbytes)
      self.bindings.append(int(cuda_mem))
      if self.engine.binding_is_input(binding):
        self.host_inputs.append(host_mem)
        self.cuda_inputs.append(cuda_mem)
      else:
        self.host_outputs.append(host_mem)
        self.cuda_outputs.append(cuda_mem)
    return self.engine.create_execution_context()

  def __del__(self):
      """Free CUDA memories."""
      del self.stream
      del self.cuda_outputs
      del self.cuda_inputs

  def predict(self, img, target=None, conf_th=0.3, raw_output = False):
      """Detect objects in the input image."""
      #print("At predict image has shape {}".format(img.shape))
      img_resized = _preprocess_trt(img, self.input_shape)
      #print("After resizing has shape {}".format(img_resized.shape))
      np.copyto(self.host_inputs[0], img_resized.ravel())

      cuda.memcpy_htod_async(
          self.cuda_inputs[0], self.host_inputs[0], self.stream)
      self.context.execute_async(
          batch_size=1,
          bindings=self.bindings,
          stream_handle=self.stream.handle)
      cuda.memcpy_dtoh_async(
          self.host_outputs[1], self.cuda_outputs[1], self.stream)
      cuda.memcpy_dtoh_async(
          self.host_outputs[0], self.cuda_outputs[0], self.stream)
      self.stream.synchronize()

      output = self.host_outputs[0]
      boxes, confs, clss = _postprocess_trt(img, output, conf_th, self.output_layout)
      #print(boxes, confs, clss)
      return self.reshape_hydra(boxes, confs, clss, target)
   
  def reshape_hydra(self, boxes, confs, clss, target):
    if target is None:
      target = "person"
    ret = []
    for i in range(len(boxes)):
      if clss[i] == self.labels.index(target):
        ret.append((boxes[i], confs[i]))
    return ret
      
      
  def load_labels(self, path_to_labels=''):
      COCO_CLASSES_LIST = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
      return COCO_CLASSES_LIST
  def download(self, model_name=""):
    pass

  def load_image_into_numpy_array(self, image, expand=False):
    if isinstance(image, str):
      image = Image.open(image)
    t = []
    t.append(time.time())
    (im_width, im_height) = image.size
    t.append(time.time())
    tmp = image.getdata()
    #print(type(tmp))
    t.append(time.time())
    tmp = np.array(tmp)
    t.append(time.time())
    np_img = tmp.reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    t.append(time.time())
    #print([t[i+1] - t[i] for i in range(len(t)-1)])
    return np.expand_dims(np_img, axis=0) if expand else np_img

  def load_image_into_numpy_array_cv(self, image, expand=False):
    t = []
    t.append(time.time())
    np_img = cv2.imread(image, 1)
    t.append(time.time())
    #print(np_img.shape)
    np_img = np_img.astype(np.uint8)
    t.append(time.time())
    #print([t[i+1] - t[i] for i in range(len(t)-1)])
    return np.expand_dims(np_img, axis=0) if expand else np_img


def _preprocess_trt(img, shape=(300, 300)):
  #print(img.shape)
  #print(shape)
  if len(img.shape) > 3:
    img = img[0]
  #print("After maybe taking a dimension off it {}".format(img.shape))
  """Preprocess an image before TRT SSD inferencing."""
  img = cv2.resize(img, shape)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img.transpose((2, 0, 1)).astype(np.float32)
  img *= (2.0/255.0)
  img -= 1.0
  return img


def _postprocess_trt(img, output, conf_th, output_layout):
  """Postprocess TRT SSD output."""
  # print("Portprocess gets image with shape {}".format(img.shape))
  if len(img.shape) > 3:
    img = img[0]
  img_h, img_w, _ = img.shape
  boxes, confs, clss = [], [], []
  for prefix in range(0, len(output), output_layout):
    #index = int(output[prefix+0])
    conf = float(output[prefix+2])
    if conf < conf_th:
        continue
    x1 = output[prefix+3]# int(output[prefix+3] * img_w)
    y1 = output[prefix+4]#int(output[prefix+4] * img_h)
    x2 = output[prefix+5]#int(output[prefix+5] * img_w)
    y2 = output[prefix+6]#int(output[prefix+6] * img_h)
    cls = int(output[prefix+1])
    boxes.append((x1, y1, x2, y2))
    confs.append(conf)
    clss.append(cls)
  #print(boxes)
  #print(confs)
  #print(clss)
  return boxes, confs, clss



if __name__ == "__main__":
  import pycuda.autoinit
  import numpy as np
  import cv2
  import time
  import os
  import urllib
  import tarfile
  import pickle
  from devices.config import *
  import sys
  # import tensorflow.contrib.tensorrt as trt
  import numpy as np
  import time
  # from tf_trt_models.detection import *
  from PIL import Image
  import tensorflow as tf
  from ssd.ssd_classes import get_cls_dict
  from ssd.ssd import TrtSSD
  from ssd.camera import add_camera_args, Camera
  from ssd.display import open_window, set_display, show_fps
  from ssd.visualization import BBoxVisualization
  model = 'ssd_mobilenet_v2_coco'
  start = int(time.time())
  frame = cv2.imread("data/6zombies.jpg")
  count = 0
  trt_ssd = Model()
  conf_th=0.3
  cls_dict = trt_ssd.load_labels()
  vis = BBoxVisualization(cls_dict)
  fps = 0.0
  tic = time.time()
  try:
    for i in range(55):
      frame = cv2.imread("data/img_{}.jpg".format(i))
      boxes_confs = trt_ssd.predict(frame, conf_th=conf_th, target="person")
      print(boxes_confs)
      boxes, confs = list(zip(*boxes_confs))
      print(boxes)
      print(confs)
      img = vis.draw_bboxes(frame, boxes, confs, [1]*len(boxes))
      img = show_fps(img, fps)
      toc = time.time()
      curr_fps = 1.0 / (toc - tic)
      fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
      tic = toc
      if True:
        cv2.imwrite('logs/IMG_{}_{}.png'.format(start, count), img)
        count += 1
      time.sleep(0.1)	
  finally:
    pass
