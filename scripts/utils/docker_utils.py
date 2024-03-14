import subprocess

import time


def get_ip_for_docker_container(container_name):
    while True:
        try:
            output = subprocess.check_output(
                ["docker", "inspect", "-f", "\"{{ .NetworkSettings.IPAddress }}\"", container_name],
                stderr=subprocess.DEVNULL)
            return output.decode("ascii").split("\"")[1]
        except subprocess.CalledProcessError:
            print("[HOST] Unable to retrieve IP address of Docker container:", container_name, "-> Try again in 2s")
            time.sleep(3)


def get_port_for_docker_container(container_name):
    while True:
        try:
            output = subprocess.check_output(
                ["docker", "inspect", "-f", "\"{{ .NetworkSettings.Ports }}\"", container_name],
                stderr=subprocess.DEVNULL)
            line = output.decode("ascii").split("\"")[1]
            if line == "map[]":
                raise subprocess.CalledProcessError(1, "")
            return int(line.split("/")[0][4:])
        except subprocess.CalledProcessError:
            print("[HOST] Unable to retrieve primary PORT of Docker container:", container_name, "-> Try again in 2s")
            time.sleep(3)
