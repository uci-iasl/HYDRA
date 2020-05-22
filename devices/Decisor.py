import time
import copy
from devices.config import *

class Decisor(object):
  def __init__(self, state):
    self.state = state

  def pipeline_update(self):
    self.clean_logs(since=time.time() - 1)

    if self.state["adapt_pipes"]:
      #print("START UPDATE PIPELINES: {}".format(self.state["mode"]))
      #print([e for e in self.state["active_pipelines"]])
      if self.state["mode"] == self.id:
        self.state["mode"] = "explore"
        self.activate_edges()
        return
      else:
        curr_means, curr_max = [], []
        curr_max_dict = {}
        ## no need to reset it: it keeps last K examples
        pipelines_log = copy.deepcopy(self.state["pipelines_log"])
        for k in pipelines_log:
          if k not in self.last_counted:
            self.last_counted[k] = {"time": 0, "value": 0}
          if pipelines_log[k] and k != self.id:
            curr_means.append((np.mean([pipelines_log[k][i][1] for i in range(len(pipelines_log[k]))]), k))
            curr_max.append((np.max([pipelines_log[k][i][1] for i in range(len(pipelines_log[k]))]), k))
            curr_max_dict[k] = np.max([pipelines_log[k][i][1] for i in range(len(pipelines_log[k]))])
            for i in range(len(pipelines_log[k])):
              if pipelines_log[k][i][0] > self.last_counted[k]["time"]:
                if pipelines_log[k][i][1] < ENERGY_SAVING_THR:
                  self.last_counted[k]["value"] = 0
                else:
                  self.last_counted[k]["value"] += 1
          else:
            curr_means.append((float("inf"), k))
            curr_max.append((float("inf"), k))
    else:
      pass
      
  def apply_policy(self, curr_means, curr_max, curr_max_dict):
    if DECISION_POLICY == "double_thr":
      min_max = min(curr_max)
      curr_best = min(curr_means)
      if min_max[0] <= DELTA_E:
        self.state["mode"] = "performance"
        self.deactivate_local()
        self.deactivate_edges_but_one(curr_best[1])
      elif DELTA_E < min_max[0] <= DELTA_L:
        self.state["mode"] = "explore_edges"
        self.deactivate_local()
        self.activate_edges()
        self.start_explore = time.time()
      elif min_max[0] > DELTA_E:
        self.state["mode"] = "explore"
        self.activate_local()
        self.activate_edges()
        self.start_explore = time.time()
    elif DECISION_POLICY == "energy_saving":
      if VERBOSE:
        print(curr_means)
      curr_best = min(curr_means)
      if curr_best[0] > 1:
        self.activate_local()
        self.activate_edges()
        self.state["mode"] = "explore"
        return
      curr_best_max = curr_max_dict[curr_best[1]]
      if VERBOSE:
        print("curr_max_dict")
      for k in curr_max_dict:
        if VERBOSE:
          print("{} : {}".format(k, curr_max_dict[k]))
      if curr_best_max < ENERGY_SAVING_THR:
        self.deactivate_local()
        self.deactivate_edges_but_one(curr_best[1])
        self.state["mode"] = "performance"
      else:
        self.activate_edges()
        self.state["mode"] = "explore_edges"
        if self.last_counted[curr_best[1]]["value"] > 5:
          self.activate_local()
          self.activate_edges()
          self.state["mode"] = "explore"
    elif DECISION_POLICY == "all_edge":
      self.activate_edges()
      self.state["mode"] = "explore_edges"

  def activate_edges(self):
    for k in self.state["active_pipelines"]:
      if "EDGE" in k:
        self.state["active_pipelines"][k] = True

  def deactivate_edges(self):
    for k in self.state["active_pipelines"]:
      if "EDGE" in k:
        self.state["active_pipelines"][k] = False

  def deactivate_edges_but_one(self, pipe):
    for k in self.state["active_pipelines"]:
      if "EDGE" in k and k != pipe:
        self.state["active_pipelines"][k] = False
  
  def activate_local(self):
    self.state["active_pipelines"][self.id] = True

  def deactivate_local(self):
    self.state["active_pipelines"][self.id] = False

  def clean_logs(self, number = -1, since= -1):
    if since > 0:
      for p in self.state["pipelines_log"]:
        for i in range(len(self.state["pipelines_log"][p])):
          try:
            if self.state["pipelines_log"][p][i][0] < since:
              del self.state["pipelines_log"][p][i]
          except Exception as e:
            pass
      return
    if number > 0:
      for p in self.state["pipelines_log"]:
        self.state["pipelines_log"][p] = self.state["pipelines_log"][p][number:]
      return
    raise ValueError("Cleaning logs needs a rule")

  def stop_module(self, module_name):
    self.modules[module_name].stop()
    
    

