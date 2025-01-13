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
config_dir = sys.argv[6]




lib_path = os.path.join(afs_path, "l1_anomaly_ae")

client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=username, password=password)

# Combine commands to keep environment settings
full_command = f"""
{env_command} && \
python3 {os.path.join(lib_path,"ci_util/smoke_script.py")} {afs_path} {config_dir}
"""

print("Running the following command:")
print(full_command)

_stdin, _stdout, _stderr = client.exec_command(full_command)

# Read output and error
while not _stdout.channel.exit_status_ready():
    time.sleep(1)  # Wait for the command to complete if necessary

stdout = _stdout.read().decode()
stderr = _stderr.read().decode()
exit_status = _stdout.channel.recv_exit_status()

# Print the output and error
print("STDOUT:")
print(stdout)
print("STDERR:")
print(stderr)

if exit_status == 0:
    print("All configuration files are compatible.")
    sys.exit(0)  # Success
else:
    print("Some configuration files are incompatible.")
    sys.exit(1)  # Failure


client.close()
