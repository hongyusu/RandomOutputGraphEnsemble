
import os
import sys
import commands
sys.path.append('/home/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes
import multiprocessing
import time
import logging
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)

current_DIR = os.getcwd() 


def singleParameterSelection(filename,graph_type,node):
  try:
    with open("../outputs/%s_%s_baselearner_parameters.mat" % (filename,graph_type)): pass
    logging.info('\t--< (node)%s,(f)%s,(type)%s' %( node,filename,graph_type))
  except:
    logging.info('\t--> (node)%s,(f)%s,(type)%s' %( node,filename,graph_type))
    os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd %s; nohup matlab -nodisplay -nosplash -nodesktop -r "run_parameter_selection '%s' '%s' '0' " > /var/tmp/tmp_%s_%s_baselearner' """ % (node,current_DIR,filename,graph_type,filename,graph_type) )
    logging.info('\t--| (node)%s,(f)%s,(type)%s' %( node,filename,graph_type))
    time.sleep(5)
  pass

def run():
  #cluster = get_free_nodes()[0]
  cluster=['dave']
  jobs=[]
  n=0

  filenames=['fpuni'] 
  n=0
  for filename in filenames:
    for graph_type in ['tree','pair']:
      node=cluster[n%len(cluster)]
      n+=1
      p=multiprocessing.Process(target=singleParameterSelection, args=(filename,graph_type,node,))
      jobs.append(p)
      time.sleep(1)
      p.start()

  for job in jobs:
    job.join()
  pass


run()


