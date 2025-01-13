import paramiko
import sys
import time
import select

env_command = "export PATH=/afs/cern.ch/user/m/mlflow/public/winter/bin:$PATH"

host = "lxplus.cern.ch"
username = sys.argv[1]
password = sys.argv[2]
afs_dir = sys.argv[3]
JOB_NAME = sys.argv[4]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=username, password=password)

# Prepare the full command
full_command = f"""
{env_command}
export PYTHONNOUSERSITE=1
echo "Starting to monitor Condor jobs with JobGroup: {JOB_NAME}"
while true; do
    # Get the list of JobStatus codes for jobs with the given JobGroup
    JOB_STATUSES=$(condor_q -constraint 'JobGroup == "{JOB_NAME}"' -format '%d\\n' JobStatus)
    # Check if there are any jobs in Idle (1) or Running (2) status
    if echo "$JOB_STATUSES" | grep -qE '^(1|2)$'; then
        # Jobs are still running or pending
        stdbuf -oL echo "========================================================================"
        stdbuf -oL echo "Jobs are still running. Waiting for 15 seconds before checking again."
        NUM_FILES=$(ls -1 {afs_dir}/{JOB_NAME}/*.out 2>/dev/null | wc -l)
        stdbuf -oL echo "$NUM_FILES jobs are completed, Current time is $(date)"
        stdbuf -oL echo "========================================================================"
        sleep 15
    else
        # No jobs are Idle or Running
        stdbuf -oL echo "All jobs have completed or failed."
        break
    fi
done
"""

# Execute the command
_stdin, _stdout, _stderr = client.exec_command(full_command, get_pty=True)

# Read the output
while True:
    # Use select to check if there is data to read
    if _stdout.channel.exit_status_ready():
        break
    if _stdout.channel.recv_ready():
        rl, wl, xl = select.select([_stdout.channel], [], [], 0.0)
        if len(rl) > 0:
            # Read data from the channel
            output = _stdout.channel.recv(1024).decode('utf-8')
            print(output, end='')  # Output to stdout for GitLab CI to capture
    time.sleep(1)  # Avoid busy waiting

# Read any remaining output
while _stdout.channel.recv_ready():
    output = _stdout.channel.recv(1024).decode('utf-8')
    print(output, end='')

# Close the client
client.close()
