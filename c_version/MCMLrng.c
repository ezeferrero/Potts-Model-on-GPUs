/*  This file is part of CUDAMCML, and adapted to run in CPU.

    CUDAMCML is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMCML is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMCML.  If not, see <http://www.gnu.org/licenses/>.*/

float rand_MWC_co(unsigned long long* x, unsigned int* a)
{
		//Generate a random number [0,1)
		*x=(*x&0xffffffffull)*(*a)+(*x>>32);
		return ((float) ((unsigned int)(*x))) / (float)0x100000000;// The typecast will truncate the x so that it is 0<=x<(2^32-1),__uint2float_rz ensures a round towards zero since 32-bit floating point cannot represent all integers that large. Dividing by 2^32 will hence yield [0,1)

}//end rand_MWC_co

float rand_MWC_oc(unsigned long long* x, unsigned int* a)
{
		//Generate a random number (0,1]
		return 1.0f-rand_MWC_co(x,a);
}//end rand_MWC_oc

int init_RNG(unsigned long long *x, unsigned int *a,
            const unsigned int n_rng, const char *safeprimes_file, unsigned long long xinit)
{
	FILE *fp = NULL;
	int fscanf_result = 0;
	unsigned int begin = 0u;
	unsigned int fora,tmp1,tmp2;

	if (strlen(safeprimes_file)==0) {
		// Try to find it in the local directory
		safeprimes_file = "safeprimes_base32.txt";
	}

	fp = fopen(safeprimes_file, "r");
	if (fp==NULL) {
		printf("Could not find the file of safeprimes (%s)! Terminating!\n", safeprimes_file);
		return 1;
	}

	fscanf_result = fscanf(fp, "%u %u %u", &begin, &tmp1, &tmp2);
	if (fscanf_result<3) {
		printf("Problem reading first safeprime from file %s\n", safeprimes_file);
		return 1;
	}

	// Here we set up a loop, using the first multiplier in the file to generate x's and c's
	// There are some restictions to these two numbers:
	// 0<=c<a and 0<=x<b, where a is the multiplier and b is the base (2^32)
	// also [x,c]=[0,0] and [b-1,a-1] are not allowed.

	//Make sure xinit is a valid seed (using the above mentioned restrictions)
	if ((xinit==0ull) | (((unsigned int)(xinit>>32))>=(begin-1)) | (((unsigned int)xinit)>=0xfffffffful)) {
		//xinit (probably) not a valid seed! (we have excluded a few unlikely exceptions)
		printf("%llu not a valid seed! Terminating!\n",xinit);
		return 1;
	}

	unsigned int i = 0;
	for (i=0; i<n_rng; i++)
	{
		fscanf_result = fscanf(fp, "%u %u %u", &fora, &tmp1, &tmp2);
		if (fscanf_result<3) {
			printf("Problem reading safeprime %d out of %d from file %s\n", i+2, n_rng+1, safeprimes_file);
			return 1;
		}

		a[i] = fora;
		x[i] = 0;
		while ((x[i]==0) | (((unsigned int)(x[i]>>32))>=(fora-1)) | (((unsigned int)x[i])>=0xfffffffful)) {
			//generate a random number
			xinit = (xinit&0xffffffffull)*(begin)+(xinit>>32);

			//calculate c and store in the upper 32 bits of x[i]
			x[i] = (unsigned int) floor((((double)((unsigned int)xinit))/(double)0x100000000)*fora);//Make sure 0<=c<a
			x[i] = x[i]<<32;

			//generate a random number and store in the lower 32 bits of x[i] (as the initial x of the generator)
			xinit = (xinit&0xffffffffull)*(begin)+(xinit>>32);//x will be 0<=x<b, where b is the base 2^32
			x[i] += (unsigned int) xinit;
		}
		//if(i<10)printf("%llu\n",x[i]);
	}
	fclose(fp);

	return 0;
}
