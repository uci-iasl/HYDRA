import os
import time
from typing import List, Dict
from collections import defaultdict
import numpy as np
import copy

def parse_netstat_i(interfaces: List[str] = None) -> Dict[str, Dict[str, str]]:
  """
  Dictionary contains:
  - for each interface
    - Iface : Interface name
      MTU   : Maximum Transmission Unit
      RX-OK : Reciving ok [bytes]
      RX-ERR: 
      RX-DRP:
      RX-OVR:
      TX-OK : 
      TX-ERR:
      TX-DRP:
      TX-OVR:
      Flg   : State of connection
  """
  out = os.popen('netstat -i').read()
  out = out.split("\n")
  lines = [e for e in out]
  header = out[1].split()
  if interfaces is not None:
    ifaces = [e[:5] for e in interfaces]
  ret = {}
  for e in lines[2:]:
    if len(e) > 0:
      tmp = e.split()
      if interfaces is None or tmp[0][:5] in ifaces:
        tmp_ret = {header[i]: tmp[i] for i in range(len(header))}
        ret[tmp[0]] = tmp_ret 
  return ret

def parse_wireless(interfaces: List[str] = None) -> Dict[str, Dict[str, str]]:
  """
  Returns, for each interface, signal strength, level, noise.
  """
  out = os.popen('cat /proc/net/wireless').read()
  out = out.split("\n")
  lines = [e for e in out]
  header = ["Iface", "Status", "Q_link", "Q_lev", "Q_noise"]
  ifaces = None
  if interfaces is not None:
    ifaces = [e[:5] for e in interfaces]
  ret = {}
  for e in lines[2:]:
    if len(e) > 0:
      tmp = e.split()
      interface = tmp[0].split(':')[0][:5]
      tmp[0] = interface
      if interfaces is None or interface in ifaces:
        tmp_ret = {header[i]: tmp[i] for i in range(len(header))}
        ret[interface] = tmp_ret
  return ret

def ss_info_tcp(skip_ssh=False):
  """
  Returns a dictionary with information about the available tcp connections.
  Info about fields: http://man7.org/linux/man-pages/man8/ss.8.html
  wscale: window scale scale factor for send,rcv
  rto: TCP retransmisison timeout [ms]
  rtt: mean/std round trip time [ms]
  ato: ack timeout [ms]
  mss: max segment size [byte]
  pmtu: path MTU value
  mss: maximum segment size [byte]
  cwnd: congestion window [byte]
  bytes_acked
  bytes_received
  segs_out: segments out
  segs_in: segments in
  data_segs_out
  data_segs_in
  lastsnd: time since the last packet was sent [ms]
  lastrcv: time since the last packet was received [ms]
  lastack: time since the last ack was received [ms]
  busy:
  rcv_rtt: 
  rcv_space: 
  rcv_ssthresh: 
  minrtt: 
  """
  st = time.time()
  out = os.popen('ss --info --tcp').read()
  out = out.split("\n")[1:]
  # print("External call {}".format(time.time()-st))
  st = time.time()
  ret = []
  conn, i = None, 0
  d = defaultdict(lambda : None)
  while i < len(out) and len(out[i]) > 0:
    if i%2==0 and out[i][:5] in ["ESTAB", "LISTE", "TIME-"]:
      d = defaultdict(lambda : None)
      split_line = out[i].split()
      d["rec_q"] = split_line[1]
      d["snd_q"] = split_line[2]
      conn = split_line[3]
      d["conn"] = conn
      d["conn_peer"] = split_line[4]
      if skip_ssh and d["conn_peer"].split(':')[1] == 'ssh':
        d = defaultdict(lambda : None)
        conn = None
    elif i%2==1 and conn is not None:
      tmp = out[i].strip().split()
      tmp1 = [e.split(':') for e in tmp]
      for e in tmp1:
        if len(e) > 1:
          d[e[0]] = e[1]
      ret.append(d)
      conn = None
    else:
      pass
      # print("Ignored line ", out[i])
    i += 1
  # print("Parsing {}".format(time.time()-st))
  return ret

def ifconfig_all():# -> Dict[str, Dict[str, str]:
  """
  For each wireless interface (starting with 'w'), returns a dictionary including all available fields in ifconfig
  """
  def RX_packets(line: List[str]) -> Dict[str, str]:
    return {"RX-OK-pck": line[2], "RX-OK-B": line[4]}
  def RX_errors(line: List[str]) -> Dict[str, str]:
    return {"RX-ERR-pck": line[2], "RX-DRP": line[4], "RX-OVR": line[6], "RX-FR": line[8]}
  def TX_packets(line: List[str]) -> Dict[str, str]:
    return {"TX-OK-pck": line[2], "TX-OK-B": line[4]}
  def TX_errors(line: List[str]) -> Dict[str, str]:
    return {"TX-ERR-pck": line[2], "TX-DRP": line[4], "TX-OVR": line[6], "TX-FR": line[8]}
  def inet(line: List[str]) -> Dict[str, str]:
    return {"ip": line[1]}
  def out_of(line: List[str]) -> Dict[str, str]:
    return {}
  switch = {'inet ': inet, "RX pa": RX_packets, "RX er": RX_errors, "TX pa": TX_packets, "TX er": TX_errors}
  switch = defaultdict(lambda : out_of, switch)
  out = os.popen('ifconfig').read()
  out = out.split("\n")
  ret = {}
  conn, i = None, 0
  while i < len(out):
    if len(out[i]) > 1:
      first_char = out[i][0]
      if first_char == "w":
        conn = out[i].split(':')[0]
        ret[conn] = {}
      elif first_char != " ":
        conn = None
      elif out[i][:8] == " "*8 and conn is not None:
        prefix = out[i][8:13]
        ret[conn].update(switch[prefix](out[i].split()))
    i += 1
  return ret
  
def switch_interface_ip(ifconfig_dict):
  ret = {}
  for e in ifconfig_dict:
    if ifconfig_dict[e]["ip"] is not None:
      ret[ifconfig_dict[e]["ip"]] = ifconfig_dict[e]
      ret[ifconfig_dict[e]["ip"]]["iface"] = e
  return ret

def test_timing(func, kargs={}):
  start = time.time()
  TRIALS = 100
  timings = []
  tmp = {}
  res = []
  for _ in range(TRIALS):
    st = time.time()
    res.append(func(**kargs))
    timings.append(time.time()-st)
  print(len(res))
  print(np.mean(timings), np.std(timings))
  
def get_netstats_separate_conn_iface(skip_ssh=False, header=None):
  wireless = parse_wireless()
  ifconfig = ifconfig_all()
  ip2iface = {}
  for e in ifconfig:
    ip2iface[ifconfig[e]["ip"]] = e
  for e in wireless:
    ifconfig[e].update(wireless[e])
  ifconfig_list = [ifconfig[e] for e in ifconfig]
  
  tcp_info = ss_info_tcp(skip_ssh=False)
  ret = {"signal": ifconfig_list, "tcp": tcp_info}
  fields, content = {}, {}
  for info_type in ['signal', 'tcp']:
    if header is None:
      fields[info_type] = sorted(ret[info_type][0])
    else:
      fields = header
    content[info_type] = [[ret[info_type][i][k] for k in fields[info_type]] for i in range(len(ret[info_type]))]
  if header is None:
    return (fields, content)
  else:
    return content

if __name__=="__main__":
  test_timing(get_netstats_separate_conn_iface)
  ret = get_netstats_separate_conn_iface()
  test_timing(get_netstats_separate_conn_iface, kargs={'header': ret[0]})
  



