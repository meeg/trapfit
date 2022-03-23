#!/usr/bin/env python
import os,sys
import csv
from datetime import datetime
import array
import math
from os.path import basename, commonprefix, splitext

from ROOT import gROOT
gROOT.SetBatch(True)
from ROOT import gStyle
from ROOT import TFile
from ROOT import TTree
from ROOT import TCanvas
from ROOT import TGraphErrors
from ROOT import gPad
from ROOT import TF1

hdus = [2,3]
data = {}
for q in hdus:
    t = array.array('d')
    r = array.array('d')
    rErr = array.array('d')
    data[q] = (t, r, rErr)

filename = sys.argv[1]
with open(filename) as csvfile:
    reader = csv.DictReader(csvfile)
    firstRow = True
    for row in reader:
        #print(row)
        runid = int(row['runid'])
        starttime = datetime.strptime(row['readoutStart'], "%Y-%m-%d %H:%M:%S")
        endtime = datetime.strptime(row['readoutEnd'], "%Y-%m-%d %H:%M:%S")
        readtime = (endtime-starttime).total_seconds()
        if firstRow:
            firsttime = starttime
            print(readtime)
            firstRow = False
        timeElapsed = (endtime-firsttime).total_seconds() + 0.5*readtime #image time is (arbitrarily) halfway between start and end

        for q in hdus:
            try:
                rate = float(row['rate'+str(q)])
                if math.isnan(rate) or rate<-0.1:
                    raise Exception
                data[q][1].append(rate)
                data[q][2].append(float(row['rateErr'+str(q)]))
                data[q][0].append(timeElapsed)
            except:
                pass

outfilename = "trapfit_"+splitext(basename(filename))[0]
outfile = TFile(outfilename+".root","recreate")
gStyle.SetOptStat(0)
gStyle.SetOptFit(1)
c = TCanvas("c","c",1200,900);
c.Print(outfilename+".pdf[")

zero = TF1("zero","0",-1e7,1e7)

objects = [] #we don't use this, but it prevents the histograms from getting garbage-collected
def doFits(dataTuple, fitfunc, name):
    t, r, rErr = dataTuple
    g = TGraphErrors(len(t), t, r, 0, rErr)
    fitfunc.SetParameter(0,t[100]*r[100])
    g.Fit(fitfunc,"S")
    g.SetTitle(name)
    g.GetXaxis().SetTitle("time elapsed [s]")
    g.GetYaxis().SetTitle("DC [e-/pix/day]")
    g.Write(name)
    res = array.array('d')
    resErr = array.array('d')
    for i, time in enumerate(t):
        res.append((r[i] - fitfunc.Eval(time))/rErr[i])
        resErr.append(1.0)
    
    gRes = TGraphErrors(len(t), t, res, 0, resErr)
    gRes.SetTitle(name+"_res")
    gRes.Write(name+"_res")
    gRes.GetXaxis().SetTitle("time elapsed [s]")
    gRes.GetYaxis().SetTitle("normalized residual")
    objects.append(g)
    objects.append(gRes)
    return g, gRes

def testFunc(fitfunc, name):
    c.Clear()
    c.Divide(1,4)

    for iHdu, q in enumerate(hdus):
        g, gRes = doFits(data[q], fitfunc, "q{0}_{1}".format(q,name))
        c.cd(1 + iHdu*2)
        gPad.SetLogy(1)
        gPad.SetLogx(1)
        g.Draw("A*")
        c.cd(2 + iHdu*2)
        gPad.SetLogx(1)
        gRes.Draw("A*")
        zero.Draw("same")

    c.cd()
    c.Print(outfilename+".pdf")

powerlaw = TF1("powerlaw","[0]*(x+[3])^[1]+[2]")
powerlaw.SetParameter(1, -1.0)
testFunc(powerlaw, "powerlaw")

powerlaw.FixParameter(1, -1.0)
testFunc(powerlaw, "powerlaw_k1")
powerlaw.ReleaseParameter(1)

powerlaw.FixParameter(2, 0.0)
testFunc(powerlaw, "powerlaw_nody")
powerlaw.ReleaseParameter(2)

c.Print(outfilename+".pdf]")
outfile.Write()
outfile.Close()
