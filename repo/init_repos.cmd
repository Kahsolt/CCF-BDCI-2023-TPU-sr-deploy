@ECHO OFF

PUSHD %~dp0

REM contest material
git clone https://github.com/sophgo/TPU-Coder-Cup

REM SR model zoo
git clone https://github.com/Coloquinte/torchSR
git clone https://github.com/Lornatang/SRCNN-PyTorch
git clone https://github.com/Lornatang/FSRCNN-PyTorch
git clone https://github.com/Lornatang/ESPCN-PyTorch

POPD

ECHO Done!
ECHO.

PAUSE
