@ECHO OFF

SET MOUNT_PATH=/workspace/code

CD /D %~dp0
docker run --name tpu-mlir --volume="%CD%":%MOUNT_PATH% --workdir %MOUNT_PATH% --memory="0" -it -d kahsolt/tpuc_dev:latest
