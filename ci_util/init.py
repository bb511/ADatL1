### This script prepares the inital setup of the codes including setting up the environment, EOS space and AFS directories.
import paramiko
import sys
import os
import time

env_command = "conda activate /afs/cern.ch/user/m/mlflow/public/winter"

host = "lxplus.cern.ch"
username = sys.argv[1]
password = sys.argv[2]
afs_path = sys.argv[3]
CI_JOB_TOKEN = sys.argv[4]
JOB_NAME = sys.argv[5]
EOS_SHARED_SPACE = sys.argv[6]
NTUPLE_EOS = sys.argv[7]


lib_path = os.path.join(afs_path, "l1_anomaly_ae")

client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=username, password=password)

# Combine commands to keep environment settings
full_command = f"""
{env_command} && \
export PYTHONNOUSERSITE=1 && \
mkdir -p {afs_path} && \
cd {afs_path} && \
git clone https://oauth2:{CI_JOB_TOKEN}@gitlab.cern.ch/cms-l1-ad/l1_anomaly_ae.git && \
cd {lib_path} && \
git checkout refactory && \
mkdir -p {EOS_SHARED_SPACE} && \
mkdir -p {os.path.join(EOS_SHARED_SPACE,"Data_file")} && \
mkdir -p {os.path.join(EOS_SHARED_SPACE,"Lite_file")} && \
mkdir -p {os.path.join(EOS_SHARED_SPACE,"Complete_file")} && \
mkdir -p {os.path.join(EOS_SHARED_SPACE,"html_report")} && \
mkdir -p {os.path.join(EOS_SHARED_SPACE,"pdf_report")} && \
python3 ./ci_util/sweep_maker.py {afs_path} {EOS_SHARED_SPACE} {NTUPLE_EOS}
"""

print("Running the following command:")
print(full_command)

_stdin, _stdout, _stderr = client.exec_command(full_command)

# Read output and error
while not _stdout.channel.exit_status_ready():
    time.sleep(1)  # Wait for the command to complete if necessary

stdout = _stdout.read().decode()
stderr = _stderr.read().decode()

# Print the output and error
print("STDOUT:")
print(stdout)
print("STDERR:")
print(stderr)

client.close()
