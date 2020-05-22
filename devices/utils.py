import random
import argparse
from devices.config import *


def rand_in(minmax):
  # print(minmax)
  tmp_r = random.random()
  # print(tmp_r)
  tmp_ret = tmp_r * (minmax[1] - minmax[0]) + minmax[0]
  # print(tmp_ret)
  return tmp_ret


def RepresentsInt(s):
  try:
    int(s)
    return True
  except ValueError:
    return False
    

def get_argparse():
  parser = argparse.ArgumentParser()
  parser.add_argument("--c", help="connection string to connect to the drone",
                      default=CONNECTION_STRING)
  # parser.add_argument("--real", help="Deactivates all the debugging conveniencies", action="store_true")
  parser.add_argument("--solo_edge", action="store_true")
  parser.add_argument("--tegra", action="store_true")
  parser.add_argument("--fly", help="Is it actually connected to a drone?", action="store_true")
  parser.add_argument("--name", help="", default="UAV01")
  parser.add_argument("--info", help="Should characterize the experiment", default="Just another experiment")
  parser.add_argument("--adaptive_fr", "-afr", action="store_true")
  parser.add_argument("--adaptive_pipes", "-apip", action="store_true")
  parser.add_argument("--move", action="store_true")
  parser.add_argument("--model", default=0)
  parser.add_argument("--input", default=INPUT_VIDEO, help="Type of input for image input - 0 is webcam, strings are files")
  parser.add_argument("--center", default=None, nargs='+')
  parser.add_argument("--radius_tot", default=35)
  parser.add_argument("--radius_me", default=7)
  parser.add_argument("--altitude", nargs=2, default=[5, 15])
  parser.add_argument("--speed", nargs=2, default=[1, 5])
  parser.add_argument("--off_n", default=0)
  parser.add_argument("--off_e", default=-3)
  parser.add_argument("--verbose", action="store_true")
  parser.add_argument("--db", type=str, help="Specify host address at which influx is running")
  parser.add_argument("--trt", action="store_true", help="Is your device TRT enabled?")

  # parser.add_argument("port")
  args = parser.parse_args()
  return args

