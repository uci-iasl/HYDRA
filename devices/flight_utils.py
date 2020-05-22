import math
from dronekit import LocationGlobal, LocationGlobalRelative
import random

def get_distance_meters(aLocation1, aLocation2):
  """
  Returns the ground distance in meters between two LocationGlobal objects.

  This method is an approximation, and will not be accurate over large distances and close to the 
  earth's poles. It comes from the ArduPilot test code: 
  https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
  """
  dlat = aLocation2.lat - aLocation1.lat
  dlong = aLocation2.lon - aLocation1.lon
  dalt_meters = aLocation2.alt - aLocation1.alt
  dlat_long_meters = (math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5)
  return math.sqrt(dlat_long_meters**2 + dalt_meters**2)


def get_location_meters(original_location, dNorth, dEast, altitude):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the 
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to 
    the current vehicle position.

    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.

    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if altitude is None:
      altitude = original_location.alt
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon, altitude)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation=LocationGlobalRelative(newlat, newlon, altitude)
    else:
        raise Exception("Invalid Location object passed")
        
    return targetlocation;


def select_out_of_my_circle(radius = 35, me_x = 10, me_y = 10, me_radius = 5):
  x = me_x
  y = me_y
  while math.sqrt((x-me_x)**2 + (y-me_y)**2) < me_radius:
    a = random.random() * 2 * math.pi
    r = radius * math.sqrt(random.random())

    x = r * math.cos(a)
    y = r * math.sin(a)
  print("In select_out_of_my_circle x, y = {}, {}".format(x,y))
  return x, y

