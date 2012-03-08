#!/bin/sh
#
# Inspired on # http://t16web.lanl.gov/Kawano/gnuplot/intro/working-e.html
#
gnuplot << EOF
set terminal postscript eps color enhanced
set output "$1.mag.eps"
set xlabel "Temperature"
set ylabel "Magnetization"
set title "Magnetization Potts"
set mxtics 5
set mytics 5
set xtics 0.025
set ytics 0.2
plot "$1" using 1:5 title "M" with lines, "$1" using 1:6 title "M^2" with lines, "$1" using 1:7 title "M^4" with lines
EOF
