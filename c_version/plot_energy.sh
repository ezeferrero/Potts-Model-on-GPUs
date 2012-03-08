#!/bin/sh
#
# Inspired on # http://t16web.lanl.gov/Kawano/gnuplot/intro/working-e.html
#
gnuplot << EOF
set terminal postscript eps color enhanced
set output "$1.energy.eps"
set xlabel "Temperature"
set ylabel "Energy"
set title "Energy Potts"
set mxtics 5
set mytics 5
set xtics 0.025
set ytics 0.2
plot "$1" using 1:2 title "E" with lines, "$1" using 1:3 title "E^2" with lines, "$1" using 1:4 title "E^4" with lines
EOF
