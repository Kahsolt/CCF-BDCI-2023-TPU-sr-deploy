@ECHO OFF

PUSHD %~dp0

REM contest material
git clone https://github.com/sophgo/TPU-Coder-Cup

REM SR model zoo
git clone https://github.com/Coloquinte/torchSR

POPD

ECHO Done!
ECHO.

PAUSE
