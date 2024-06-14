#!/bin/bash

# Define the folder to copy and the list of IPs
FOLDER_TO_COPY="$(pwd)/Tuning1"
IPS=("192.168.25.180" "192.168.25.179" "192.168.25.178" "192.168.25.177" "192.168.25.176" "192.168.25.175" "192.168.25.174" "192.168.25.173" "192.168.25.172" "192.168.25.171" "192.168.25.170" "192.168.25.169") #spaces as separation

#IPS=("192.168.25.173")


# Define the user and password
USER="labuser"
PASSWORD="<enter password here>" #TODO: enter password here
DESTINATION_PATH="/home/labuser/optuna2"

# Loop over each IP, create the destination path, and copy the folder using scp
for IP in "${IPS[@]}"; do
  echo "Creating destination path for $IP"
  sshpass -p "$PASSWORD" ssh "$USER@$IP" "mkdir -p $DESTINATION_PATH"
  echo "copying to $IP..."
  sshpass -p "$PASSWORD" scp -r "$FOLDER_TO_COPY" "$USER@$IP:$DESTINATION_PATH"
  
  #sshpass -p "$PASSWORD" scp -r "$FOLDER_TO_COPY/run_worker.py" "$USER@$IP:$DESTINATION_PATH/Tuning1"
done

echo "All copies completed."
