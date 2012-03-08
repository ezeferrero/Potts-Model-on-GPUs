#!/usr/bin/python

import string, sys, re
import collections, itertools

#
# Parse & summarize
#

log_version = ""
# holds the list of measurements declared in the header
header_measurements = []
# for each method,measurement, we have Values, namely min, sum, max and samples
Values = collections.namedtuple('Values', 'min sum max samples')
# stat[measurement][method] dictionary is a Values tuple holding min, sum, max and samples
stat = {}

for line in file(sys.argv[1], "r") :
	line = line.rstrip() # remove trailing NL
	if line[0]=="#" :
		words = line.split(' ')
		if words[1]=="CUDA_PROFILE_LOG_VERSION" :
			log_version = words[2]
		continue
	if line.find('method,')==0 : # is header
		header_measurements = line.split(',')
		continue
	assert (line.find('method=[')==0), "Some header was not processed"
	
	i = 0
	while ""<line :
		if i==0 : # method = [ name ]
			parts = re.split('(\w+)=\[\s*(\w+)\s*\]\s*', line, 1)
		else : # measure = [ value ]
			parts = re.split('(\w+)=\[\s*(\d+\.\d*)\s*\]\s*', line, 1)
		key = parts[1]
		value = parts[2]
		line = parts[3] # the rest to be processed
		if i==0 :
			method = value
			if (not method in stat) :
				stat[method] = {}
		else :
			measurement = key
			value = float(value)
			if (not measurement in stat[method]) :
				stat[method][measurement] = Values(min=value, sum=value, max=value, samples=1)
			else :
				old = stat[method][measurement]
				stat[method][measurement] = Values(min(value,old.min), 
								old.sum+value, max(value,old.max), old.samples+1)
		i += 1

# There must be something out there
assert 0<len(stat), "Empty stats"

# The amount of samples for all method and measurement should be equal
for method in stat.iterkeys() :
	assert (max([stat[method][measurement].samples for measurement in stat[method].iterkeys()]) ==
		min([stat[method][measurement].samples for measurement in stat[method].iterkeys()])), "There's a line for method "+method+" with less measurements fields"

# From now on you could safely use the samples number for the first measurement:
#	stat[method][stat[method].keys()[0]].samples

# min<=avg<=max for every measurement in every method
for method in stat.iterkeys() :
	samples = stat[method][stat[method].keys()[0]].samples
	for measurement in stat[method] :
		assert stat[method][measurement].min<=0.1e-10+stat[method][measurement].sum/samples, "min>avg in "+method+" "+measurement
		assert stat[method][measurement].sum/samples-0.1e-10<=stat[method][measurement].max, "avg>max in "+method+" "+measurement

#
# Output
#

# method breakdown
print '{0:>30} {1:^6} {2:^4}'.format('method', 'calls', '%')
total = sum([stat[method][stat[method].keys()[0]].samples for method in stat.iterkeys()])

for method in stat.iterkeys() :
	samples = stat[method][stat[method].keys()[0]].samples
	print '{0:>30} {1:>6d} {2: >.3f}'.format(method, samples, samples/float(total))

# method measurements
for method in stat.iterkeys() :
	print
	print '{0:>30} {1:^7} {2:^7} {3:^7}'.format(method, "Min", "Avg", "Max")
	samples = stat[method][stat[method].keys()[0]].samples
	assert 0<samples, "Empty measures for "+method
	for measurement in stat[method] :
		print '{0:>30} {1:03.3f} {2:03.3f} {3:03.3f}'.format(measurement,
			stat[method][measurement].min,
			stat[method][measurement].sum/samples,
			stat[method][measurement].max)


#print "DEBUG>", "CUDA_PROFILE_LOG_VERSION: ", log_version
#print "DEBUG>", "Measurements: ", header_measurements
#print "DEBUG>", stat
