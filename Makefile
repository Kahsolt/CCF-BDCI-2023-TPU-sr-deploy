all:
	g++ -I/opt/sophon/libsophon-current/include \
	  -I/opt/sophon/sophon-sail/include \
	  -I/opt/sophon/sophon-sail/include/sail \
	  -I/opt/sophon/sophon-sail/include/sail/spdlog \
	  -L/opt/sophon/libsophon-current/lib \
	  -L/opt/sophon/sophon-sail/lib \
	  run_bmodel.cpp \
	  -o run_bmodel.exe
