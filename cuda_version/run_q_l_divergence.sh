#/bin/sh
export CUDA_PROFILE="1"
export CUDA_PROFILE_CONFIG="profile_divergence"
export LD_LIBRARY_PATH="/opt/cuda/lib64"
for q in 6 8 9 12 15 24 48 96 192
do
	t_crit=`echo "scale=10; 1/l(1+sqrt($q))" | bc -l`
	for ((l=512; l<=32768; l=l*2))
	do
		echo -n "$q $l "
		sed "s/_REPLACE_Q_/$q/;s/_REPLACE_L_/$l/;s/_REPLACE_T_/$t_crit/" Makefile.parameters > Makefile.tmp
		make -f Makefile.tmp > /dev/null 2>&1
		./potts3 > /dev/null 2>&1
		awk '/updateCUDA/ { n += 1; s += $17/$14 } END { print s/n }' cuda_profile_0.log
	done
	# Two-line seprator for GNUPlot
	echo " "
	echo " "
done
export CUDA_PROFILE="0"
rm Makefile.tmp
