import os
import numpy as np
INITIAL_FRAME_RATE = 5
JPEG_QUALITY = 20


EXPLORE_INTERVAL=1
PIPELINE_UPDATE_PRECISION=0.1

Q_READ_TIMEOUT = 10000
INPUT_VIDEO = 0 #"data/VIDEOS/UAV123_person1.mp4" #0#"test_video40.avi"#0#"/Users/davide/Google Drive/University/PhD/19_sigcom_w/GOPR9840.MP4"#0#"test_video40.avi"
SAVE_IMG = True
SHOW_IMG = False
DAEMONIAC_THREADS = True
CONNECTION_STRING = "/dev/ttyUSB0"#"127.0.0.1:14550"

# Variables for detection -> movement
Q_DIM = 1
MAX_IMG_QUEUE = 1
X_CUTOFF = 0.1
Y_CUTOFF = 0.1
AREA_CUTOFF = 1000
FORWARD = 1
BACKWARD = -1
RIGHT = 1
LEFT = -1
UP = 1
DOWN = -1
TARGET_AREA = 10000
CENTER = np.array((0.5, 0.5, TARGET_AREA))

ALPHA = 0.1
MOVEMENT_DURATION = 0.4

BUFF_SIZE = 1024
MODEL_PATH = "models"

MESSAGE = "Ohbellaciao-{}"
POS_MESSAGE = "Pos;{};{};{};{}"
Q_READ_TIMEOUT = 10000
DEVICE_STATE_UPDATE = 0.5
MIN_MOVING_TIME = 0.2
POS_INTERVAL = 1

UDP_DISCOVERY_PORT = 5006
INITIAL_TCP_PORT = 12001
POS_PORT = 16543
BUFF_SIZE = 1024
JPEG = True

LOGS_PATH = "logs"
BOOKKEEPER_path = os.path.join(LOGS_PATH, "_tenboom.csv")
FLIGHTLOG_NAME_path = os.path.join(LOGS_PATH, "FlightLog_{}.csv")
DEVICE_LOG_path = os.path.join(LOGS_PATH, "bubi_{}.csv")
BOOKKEEPER = "_tenboom.csv"
FLIGHTLOG_NAME = "FlightLog_{}.csv"
DEVICE_LOG = "bubi_{}.csv"

DELTA_E = 0.3
DELTA_L = 0.7

models = ["ssd_mobilenet_v2_coco",
	        "ssd_mobilenet_v1_coco_2018_01_28",
          "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03",
          "ssd_mobilenet_v2_coco_2018_03_29",
          "ssdlite_mobilenet_v2_coco_2018_05_09",
          "ssd_mobilenet_v3_small_coco_2019_08_14",
          "ssdlite_mobilenet_v2_coco_FP32_50_trt.pb"]

DECISION_POLICY = "all_edge"
ENERGY_SAVING_THR = 0.25
VERBOSE = False
CLIENT_INITIAL_ALTITUDE = 7
EDGE_ALTITUDE = 15

DBNAME = 'demo'
COUNT = 0
TRT = True
