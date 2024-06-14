#!/bin/bash

# Define the list of IPs
IPS=(
    "192.168.25.180"
    "192.168.25.179"
    "192.168.25.178"
    "192.168.25.177"
    "192.168.25.176"
    "192.168.25.175"
    "192.168.25.174"
    "192.168.25.173"
    "192.168.25.172"
    "192.168.25.171"
    "192.168.25.170"
    "192.168.25.169"
)

# Define the user and password
USER="labuser"
PASSWORD="<enter password here>" #TODO: enter password here

# Loop over each IP and open a new terminal tab to execute the commands
for IP in "${IPS[@]}"; do
    LAST_THREE_NUMBERS=$(echo $IP | awk -F. '{print $2"."$3"."$4}')
    gnome-terminal --tab --title="$LAST_THREE_NUMBERS" -- bash -c "
        sshpass -p '$PASSWORD' ssh $USER@$IP 'cd optuna2/Tuning1 && python3 run_worker.py';
        exec bash"
done

echo "All commands executed in new terminal tabs."

