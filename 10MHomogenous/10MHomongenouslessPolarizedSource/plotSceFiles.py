#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:14:16 2019

@author: eeckert
"""
import os
import obspy
sacDir = "./sw4output"
outputDir = "./processed/sacSeismograms"
os.makedirs(outputDir,exist_ok=True)
stx = obspy.read(sacDir+"/*.x")
sty = obspy.read(sacDir+"/*.y")
stz = obspy.read(sacDir+"/*.z")
for i in range(len(stx)):
    #plot all x
    stx[i].plot(outfile=outputDir+"/"+"line"+stx[i].id+".png")
for i in range(len(sty)):
    #plot all x
    sty[i].plot(outfile=outputDir+"/"+"line"+stx[i].id+".png")
for i in range(len(stz)):
    #plot all x
    stz[i].plot(outfile=outputDir+"/"+"line"+stx[i].id+".png")