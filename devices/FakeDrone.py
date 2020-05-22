'''
Created on Dec 15, 2017

@author: dcallega
'''
import threading
import time
from dronekit import LocationGlobal
import argparse

class FakeDrone(threading.Thread):
    '''
    classdocs
    '''
    def __init__(self, connection_string = 'tcp:127.0.0.1:5763', pos=LocationGlobal(33.643118, -117.826223, 0)):
      '''
      Constructor
      '''
      threading.Thread.__init__(self)
      self.connection_string = connection_string
      print("Connected with " + self.connection_string)
      if isinstance(pos, str):
        coords = [float(e) for e in pos.split(',')]
        pos = LocationGlobal(*coords)
      self.location = pos
        
    def set_movement(self, lr, ud, fb, duration=0.5, speed=200):
      # print("Moving " + " ".join([str(e) for e in [lr, ud, fb, duration, speed]]))
      time.sleep(duration)

    def move(self, *args, **kargs):
      pass
      # print("You called Move")
        
    def set_takeoff(self, really=True):
      print("Set takeoff " + str(really))
        
    def set_up(self, really=True):
      print("Set up " + str(really))
        
    def set_mode(self, mode):
      print("Set mode " + mode)
    
    def set_land(self, really=True):
      print("Set land " + str(really))
        
    def stop(self):
      print("STOP")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gps", type=str, help="Specify GPS coord of the stationary edge", default=None)
  args = parser.parse_args()
  v = FakeDrone(args.gps)
  print(v.location)
  print(v.location.lat, v.location.lon, v.location.alt)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
