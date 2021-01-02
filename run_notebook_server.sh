#!/bin/bash
# Run jupyter notebook server accessible for remote use
# --no-browser = don't pop up a browser in VM
# port 8889 is forwarded to port 2201 on host machine with IP 192.168.81.146
# --ip=0.0.0.0 makes server able to be accessed from host machine's IP
jupyter-notebook --no-browser --port=8889 --ip=0.0.0.0
