CC = gcc
CFLAGS = -O3 -std=c11 -Wall -Wextra

TARGETS = gen_goppa gen_random gen_xorfree

all: $(TARGETS)

gen_goppa: gen_goppa.c
	$(CC) $(CFLAGS) -o $@ $<

gen_random: gen_random.c
	$(CC) $(CFLAGS) -o $@ $<

gen_xorfree: gen_random_xorfree.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean