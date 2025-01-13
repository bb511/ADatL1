import paramiko
import sys
import os
import time

host = "lxplus.cern.ch"
username = sys.argv[1]
password = sys.argv[2]
afs_path = sys.argv[3]

client = paramiko.client.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=username, password=password)

full_command = f"""
for sub_file in {afs_path}/*.sub; do \\
    condor_submit "$sub_file"; \\
done
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
