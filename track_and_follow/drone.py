import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from queue import LifoQueue as Queue
import devices.data_types
from devices.config import *
from threading import Thread
import argparse
import numpy as np
import copy
from devices.flight_utils import *
from devices.Device import Device
from devices.utils import rand_in


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
  parser.add_argument("--model", default=2)
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

  # parser.add_argument("port")
  args = parser.parse_args()
  return args


"""
Moves in a cilinder defined by:
 - initial position as center of cilinder
 - radius: args.radius_tot
 - hight: in alt_min, alt_max
 - speed: between speed_min, speed_max

"""

if __name__ == "__main__":
  args = get_argparse()
  START = time.time()
  ALTITUDE = [float(e) for e in args.altitude]
  SPEED = [float(e) for e in args.speed]

  RADIUS_TOT, RADIUS_ME = float(args.radius_tot), float(args.radius_me)
  offset_N, offset_E = args.off_n, args.off_e
  with open(BOOKKEEPER_path, "a+") as f:
    f.write(",".join([FLIGHTLOG_NAME.format(START), DEVICE_LOG.format(START), args.info]))
    f.write("\n")

  #v = log = None
  d = Device(args.name, START, args.solo_edge, args.adaptive_fr, args.adaptive_pipes, is_nano=args.tegra,
             model=models[int(args.model)], ismac=False, verbose=args.verbose,
             input_type =args.input, conn_string=args.c, db_host=args.db)
  #v = d.drone
  #print("Flight Log Start")
  #log = FlightLog(v, logtime=0.1, start_time=START)
  #log.start()

  try:
    #v.arm_and_takeoff(ALTITUDE[0])
    d.start()
    while True:
      time.sleep(1)
      print("d.state[img_out_q]", d.state["img_out_q"])
      print("Active pipelines", d.state["active_pipelines"])
  except ValueError as e:
    raise e
  except KeyboardInterrupt as ki:
    pass
  except Exception as e:
    raise e
  finally:
    try:
      print("STOPPING")
      d.stop()
      log.flag("LAND")
      log.stop()
      v.quit_manual_RTL()
    except Exception as e:
      print(e)
    finally:
      d.stop()
