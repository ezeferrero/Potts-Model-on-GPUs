#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "gmp.h"

/*
 * Taken from "CUDAMCML. User manual and implementation notes"
 * Section C.1.3: Finding suitable multipliers
 */

#define FRAME 512
#define AMOUNT_SAFE_PRIMES (FRAME*(FRAME/2)+1)

static int isprime(mpz_t n);

int main(void)
{
	FILE *file = NULL;
	mpz_t n1, n2, a;
	unsigned long long i = 0;
	mpz_init(n1);
	mpz_init(n2);
	mpz_init(a);
	mpz_set_str(a, "4294967118", 0);
	file = fopen("safeprimes_base32.txt", "w");
	assert(file!=NULL);
	while (i<AMOUNT_SAFE_PRIMES) {
		mpz_mul_2exp(n2, a, 32);
		mpz_sub_ui(n2, n2, 1lu);
		if (isprime(n2)) {
			mpz_sub_ui(n1, n2, 1lu);
			mpz_fdiv_q_2exp(n1, n1, 1lu);
			if (isprime(n1)) {
				/* We have found our safeprime, calculate a and print to file */
				mpz_out_str(file, 10, a);
				fprintf(file, " ");
				mpz_out_str(file, 10, n2);
				fprintf(file, " ");
				mpz_out_str(file, 10, n1);
				fprintf(file, "\n");
				printf("%llu\n", i++);
			}
		}
		mpz_sub_ui(a, a, 1lu);
	}
	fclose(file);
	exit(0);
}

static int isprime(mpz_t n)
{
	int i;
	int class;
	int alist[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};
	for (i = 0; i<13; i++) {
		class = mpz_probab_prime_p(n, alist[i]);
		if (class==0) {
			return 0;
		}
	}

	return 1;
}
