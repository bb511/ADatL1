import sys
import os
import glob

afs_path = sys.argv[1]
config_dir = sys.argv[2]
job_id = sys.argv[3]
lib_path = os.path.join(afs_path, "l1_anomaly_ae")
sys.path.append(lib_path)


if __name__ == "__main__":
    
    config_file_list = glob.glob(os.path.join(config_dir,"*.yml"))
    print("Number of Config Files found",{len(config_file_list)})
    
    
    for file in config_file_list:
        file_name = file.split("/")[-1].split(".yml")[0]

        job_path = os.path.join(afs_path,f"{file_name}.sh")
        # The job script needs to be cleaned later with a proper conda install
        job_script =f"""#!/bin/bash

export PATH=/afs/cern.ch/user/m/mlflow/public/winter/bin:$PATH

export PYTHONNOUSERSITE=1

python3 {os.path.join(lib_path,"trainer.py")} --config_path {file} --experiment_name {job_id}

wait
"""
        with open(job_path, 'w') as job_file:
            job_file.write(job_script)


        submit_path = os.path.join(afs_path,f"{file_name}.sub")
        submit_file =f"""executable              = {job_path}
output                  = {afs_path}/$(ClusterId).$(ProcId).out
error                   = {afs_path}/$(ClusterId).$(ProcId).err

requirements = (OpSysAndVer =?= "AlmaLinux9")
+JobGroup = "{job_id}"
+MaxRuntime = 7200
request_CPUs = 8
queue
"""
#         submit_file =f"""executable              = {job_path} # This is what should be used
# output                  = {afs_path}/$(ClusterId).$(ProcId).out
# error                   = {afs_path}/$(ClusterId).$(ProcId).err

# requirements = (OpSysAndVer =?= "AlmaLinux9")
# +JobGroup = "{job_id}"
# +MaxRuntime = 7200
# request_GPUs = 1
# request_CPUs = 8
# queue
# """

        
        with open(submit_path, 'w') as sub_file:
            sub_file.write(submit_file)
