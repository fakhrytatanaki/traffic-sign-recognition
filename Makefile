CC=clang++
CFLAGS=--std=c++11 -Wall `pkg-config opencv4 --cflags --libs`
main:
	$(CC) $(CFLAGS) main.cpp

debug:
	$(CC) $(CFLAGS) -g main.cpp 

clean: a.out
	rm a.out
