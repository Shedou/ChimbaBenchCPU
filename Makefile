# Native build.
CC ?= cc
CFLAGS ?= -O3 -fopenmp -Wall
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
INSTALL ?= install

.PHONY: all install clean

all: ChimbaBenchCPU

ChimbaBenchCPU: $(wildcard src/*.c)
	$(CC) $(CFLAGS) -o $@ src/ChimbaBenchCPU.c

install: ChimbaBenchCPU
	$(INSTALL) -d $(DESTDIR)$(BINDIR)
	$(INSTALL) -m 755 ChimbaBenchCPU $(DESTDIR)$(BINDIR)/ChimbaBenchCPU

clean:
	$(RM) ChimbaBenchCPU
