
import os
import sys
import commands
sys.path.append('/home/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes
import multiprocessing
import time
import logging
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)



def singleMAVAMMlearner(filename,graph_type,t,node):
  try:
    with open("../outputs/%s_%s_%s_AMMlearner.mat" % (filename,graph_type,t)): pass
    logging.info('\t--< (node)%s,(f)%s,(type)%s,(t)%s' %( node,filename,graph_type,t))
  except:
    logging.info('\t--> (node)%s,(f)%s,(type)%s,(t)%s' %( node,filename,graph_type,t))
    os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /home/group/urenzyme/workspace/sop/mlj_ensemblemmcrf/tree_inference_scripts/; nohup matlab -nodisplay -nosplash -nodesktop -r "run_MAV_AMMlearner '%s' '%s' '%s'" > /var/tmp/tmp_%s_%s_%s_MAVAMMlearner' """ % (node,filename,graph_type,t,filename,graph_type,t) )
    logging.info('\t--| (node)%s,(f)%s,(type)%s,(t)%s' %( node,filename,graph_type,t))
    time.sleep(5)
  pass

def run():
  cluster = get_free_nodes()[0]
  #cluster = ['dave']
  jobs=[]
  n=0
  is_main_run=0.01

  filenames=['emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50','toy100'] 
  n=0
  for filename in filenames:
    for graph_type in ['pair','tree']:
      for t in range(180):
        para_t="%d" % (t+1)
        node=cluster[n%len(cluster)]
        n+=1
        p=multiprocessing.Process(target=singleMAVAMMlearner, args=(filename,graph_type,para_t,node,))
        jobs.append(p)
        p.start()
        time.sleep(1*is_main_run)
      time.sleep(60*is_main_run)

  for job in jobs:
    job.join()
  pass


run()


