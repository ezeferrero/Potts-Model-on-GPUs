#/bin/sh
export CUDA_PROFILE="1"
export CUDA_PROFILE_CONFIG="profile_null"
export LD_LIBRARY_PATH="/opt/cuda/lib64"
# Fix q=9
for q in 9
do
	t_crit=`echo "scale=10; 1/l(1+sqrt($q))" | bc -l`
	for ((l=512; l<=32768; l=l*2))
	do
		echo -n "$q $l "
		sed "s/_REPLACE_Q_/$q/;s/_REPLACE_L_/$l/;s/_REPLACE_T_/$t_crit/" Makefile.parameters > Makefile.tmp
		make -f Makefile.tmp > /dev/null 2>&1
		./potts3 > /dev/null 2>&1
		awk '/updateCUDA/ { n += 1; spinflip = $5/("'"$l"'"^2/2)*1000; s += spinflip; s2 += spinflip^2 } \
		     END {	avg = s/n; \
				avg2 = s2/n; \
				stddev = sqrt(avg2 - avg^2); \
				print avg, stddev }' cuda_profile_0.log
	done
	# Two-line seprator for GNUPlot
	echo " "
	echo " "
done
export CUDA_PROFILE="0"
rm Makefile.tmp
