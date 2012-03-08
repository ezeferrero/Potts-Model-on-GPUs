TARGET=find_safe_primes
CC=gcc
CFLAGS+=-O3 -std=c99 -Werror -Wall -Wextra -pedantic -Wdeclaration-after-statement -Wbad-function-cast -Wcast-qual -Wstrict-prototypes -Wmissing-declarations -Wmissing-prototypes -Wno-unused-parameter
LDFLAGS=-lgmp

SOURCES=$(shell echo *.c)
OBJECTS=$(SOURCES:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJECTS) .depend *~
	
.depend: *.[ch]
	$(CC) -MM $(LDFLAGS) $(SOURCES) > .depend

-include .depend

.PHONY: clean all
