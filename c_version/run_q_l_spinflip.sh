#/bin/sh
# Fix q=9
for q in 9
do
	t_crit=`echo "scale=10; 1/l(1+sqrt($q))" | bc -l`
	for ((l=512; l<=32768; l=l*2))
	do
		echo -n "$q $l "
		sed "s/_REPLACE_Q_/$q/;s/_REPLACE_L_/$l/;s/_REPLACE_T_/$t_crit/" Makefile.parameters > Makefile.tmp
		make -f Makefile.tmp > /dev/null 2>&1
		./potts3-cpu | \
			awk '/PROFILE/ { n += 1; spinflip = $2; s += spinflip; s2 += spinflip^2 } \
			END {	avg = s/n; \
				avg2 = s2/n; \
				stddev = sqrt(avg2 - avg^2); \
				print avg, stddev }'
	done
	echo " "
	echo " "
done
rm Makefile.tmp
