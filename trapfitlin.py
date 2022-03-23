#!/usr/bin/env python
import numpy as np
from scipy import integrate, LowLevelCallable
from numpy import linalg
from numpy.polynomial import Polynomial
from collections import defaultdict
import os,sys, ctypes
import csv
from datetime import datetime
import array
import math
from os.path import basename, commonprefix, splitext

from ROOT import gROOT
gROOT.SetBatch(True)
from ROOT import Math
from ROOT import gStyle
from ROOT import TFile
from ROOT import TTree
from ROOT import TCanvas
from ROOT import TGraphErrors
from ROOT import gPad
from ROOT import TF1

np.set_printoptions(linewidth=200)

hdus = [2,3]
degrees = range(1,15)

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

outfilename = "trapfitlin_"+splitext(basename(filename))[0]
outfile = TFile(outfilename+".root","recreate")
gStyle.SetOptStat(0)
gStyle.SetOptFit(1)
c = TCanvas("c","c",1200,900);
c.Print(outfilename+".pdf[")
outfile.cd()

zero = TF1("zero","0",-1e7,1e7)

lib = ctypes.CDLL(os.path.abspath('trapfit.so'))

lib.single_integrand.restype = ctypes.c_double
lib.single_integrand.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))
componentfunc = LowLevelCallable(lib.single_integrand)
def dc_component(t_arr, deltat, degree):
    y = [86400*integrate.quad(componentfunc, 0.0, 1.0, args=(t-deltat, degree))[0] for t in t_arr]
    return y

#from minuit pol1 fit
initialpars = {}
initialpars[2] = array.array('d', [0.58, 4.37e2, 2.51e3, -2.95e3])
initialpars[3] = array.array('d', [0.66, 3.51e2, 2.65e3, -3.18e3])

components = defaultdict(list)
for iHdu, q in enumerate(hdus):
    t_arr, r_arr, rErr_arr = data[q]
    deltat =  initialpars[q][1]
    for degree in range(max(degrees)+1):
        components[q].append(dc_component(t_arr, deltat, degree))

def linFit(dataTuple, deltat, maxdeg):
    t_arr, r_arr, rErr_arr = dataTuple
    y_arrs = []
    y_arrs.append(np.ones_like(t_arr)) # dc_eq

    # if you want to compute the polynomial components as-needed:
    #for degree in range(0,maxdeg+1):
    #    y_arrs.append(dc_component(t_arr, deltat, degree))
    y_arrs.extend(components[q][:maxdeg+1])
    A = np.stack(y_arrs).T
    B = np.array(r_arr)
    rootW = np.power(np.array(rErr_arr),-1)
    Aw = A * rootW[:, np.newaxis]
    Bw = B * rootW
    # this is equivalent but doesn't give us all the info we want:
    #X = linalg.lstsq(Aw, Bw)
    #bestfit = X[0]
    cov = linalg.inv(np.matmul(Aw.T, Aw))
    bestfit = np.matmul(np.matmul(cov, Aw.T), Bw)
    chi2 = np.power(linalg.norm(np.matmul(Aw, bestfit) - Bw),2)
    rchi2 = chi2/(len(B) - len(bestfit))
    return bestfit, cov, chi2, rchi2

nplot = 200
zeroArr = array.array('d',[0]*nplot)
f0 = TF1("pol0","0",0,1.0)

linfits = defaultdict(list)
linpols = defaultdict(list)
linpolgraphs = defaultdict(list)
yErrArrs = defaultdict(list)
for iHdu, q in enumerate(hdus):
    for maxdeg in degrees:
        bestfit, cov, chi2, rchi2 = linFit(data[q], initialpars[q][1], maxdeg)
        print("chi2 %f, reduced chi2 %f, params %s" % (chi2, rchi2, str(bestfit)))

        # drop dc_eq from the polynomial parameters and the covariance matrix
        # (this is equivalent to fixing it at its best-fit value)
        bestfit = bestfit[1:]
        cov = cov[1:,1:]
        linfits[q].append((bestfit,cov))

        newpol = TF1("linpol%d_%d"%(maxdeg,q),"pol%d(0)"%(maxdeg),0,1.0)
        newpol.SetParameters(array.array('d',bestfit))
        linpols[q].append(newpol)

        density = Polynomial(bestfit)
        eArr = array.array('d',np.linspace(0.0,1.0,nplot))
        yErrArr = array.array('d')
        yArr = array.array('d')
        for iE, E in enumerate(eArr):
            yArr.append(density(E))
            coeffVec = np.power(E, np.arange(maxdeg+1))
            yErrArr.append(np.matmul(np.matmul(coeffVec, cov), coeffVec))
        yErrArrs[q].append(yErrArr)
        grcov = TGraphErrors(nplot, eArr, yArr, zeroArr, yErrArr)
        linpolgraphs[q].append(grcov)

for q in hdus:
    c.Clear()
    c.SetLogx(0)
    c.SetLogy(0)
    for iPol, pol in enumerate(linpols[q]):
        if iPol==0:
            pol.Draw()
            pol.GetXaxis().SetRangeUser(0.3,0.8)
            pol.GetYaxis().SetRangeUser(0,1500)
        else:
            pol.Draw("same")
        pol.SetLineColor(iPol+2)
        grcov = linpolgraphs[q][iPol]
        grcov.SetFillColor(iPol+2)
        grcov.SetFillStyle(3005)
        grcov.Draw("3")
    c.Print(outfilename+".pdf")

for q in hdus:
    for iDeg in range(len(degrees)-1):
        prevfit,cov = linfits[q][iDeg]
        thisfit,nextcov = linfits[q][iDeg+1]
        degree = degrees[iDeg+1]

        deltafit = thisfit.copy()
        deltafit[:len(prevfit)] -= prevfit
        deltapol = TF1("deltapol%d_%d"%(degree,q),"pol%d(0)"%(degree),0,1.0)
        deltapol.SetParameters(array.array('d',deltafit))
        yArr = array.array('d', [deltapol.Eval(e) for e in eArr])

        prevgr = TGraphErrors(nplot, eArr, zeroArr, zeroArr, yErrArrs[q][iDeg])
        thisgr = TGraphErrors(nplot, eArr, yArr, zeroArr, yErrArrs[q][iDeg+1])
        deltapol.Draw()
        f0.Draw("same")
        f0.SetLineColor(2)
        deltapol.SetLineColor(4)
        prevgr.Draw("same3")
        thisgr.Draw("same3")
        thisgr.SetFillColor(4)
        thisgr.SetFillStyle(3005)
        prevgr.SetFillColor(2)
        prevgr.SetFillStyle(3004)
        deltapol.GetXaxis().SetRangeUser(0.35,0.65)
        deltapol.GetYaxis().SetRangeUser(-10,10)
        c.Print(outfilename+".pdf")


c.Print(outfilename+".pdf]")
outfile.Write()
outfile.Close()

