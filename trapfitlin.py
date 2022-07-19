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
from ROOT import TGraphErrors, TGraph
from ROOT import gPad
from ROOT import TF1

np.set_printoptions(linewidth=200)

os.system("gcc -shared -fPIC -o trapfit.so trapfit.c")

hdus = [2,3]
degrees = range(1,15)

objects = [] #we don't use this, but it prevents the histograms from getting garbage-collected

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

def single_integrand_numpy(E, t, degree):
    """
    fully vectorized function, for use with fixed_quad
    """
    density = np.power(E, degree)
    temp = 170 # K
    kt = 8.62e-5 * temp # eV
    prefactor = 1e-15 * 1.6e21

    decayrate = prefactor * pow(temp,2) * np.exp(-E/kt) # s^-1
    return 86400 * density * decayrate * np.exp(np.outer(-t, decayrate))

lib = ctypes.CDLL(os.path.abspath('trapfit.so'))
lib.single_integrand.restype = ctypes.c_double
lib.single_integrand.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))
componentfunc = LowLevelCallable(lib.single_integrand)

#minE, maxE = [0.0, 1.0]
#minE, maxE = [0.3, 0.7]
minE, maxE = [0.4, 0.6]

def do_integral_quad(t, degree):
    # epsabs is 1.49e-8 by default - that's the max error of the integral
    # within the limit set by epsabs, the integrator sometimes randomly gets lazy:
    # you can get weird discontinuous values for very specific values of t
    # you'll see weird spikes on the fitted DC vs. t curve
    # we actually seem to get more stable results by making epsabs relatively large

    return integrate.quad(componentfunc, minE, maxE, args=(t, degree), epsabs=1e-6)

def do_integral_fixed(t, degree):
    return integrate.fixed_quad(single_integrand_numpy, minE, maxE, args=(t, degree), n=10000)
    #return integrate.quadrature(single_integrand_numpy, minE, maxE, args=(t, degree))

def dc_component(t_arr, deltat, degree):
    y = []
    for t in t_arr:
        result = do_integral_quad(t-deltat, degree)
        #print(result[1])
        y.append(result[0])
    y = np.array(y)
    if (np.max(y[1:] - y[:-1])) > 0:
        # the DC vs. T curve should be monotonically decreasing, so any increase is due to integrator error - print some debug
        tmax = np.argmax(y[1:] - y[:-1])
        print(degree, t_arr[tmax], y[tmax], y[tmax+1], y[tmax+1]-y[tmax])
        print(do_integral_quad(t_arr[tmax]-deltat, degree))
        print(do_integral_fixed(t_arr[tmax]-deltat, degree))
        print(do_integral_quad(t_arr[tmax+1]-deltat, degree))
        print(do_integral_fixed(t_arr[tmax+1]-deltat, degree))
    return np.array(y)

    #return do_integral_fixed(np.array(t_arr)-deltat, degree)

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
    fitfunc = np.matmul(A, bestfit)
    resids = Bw - np.matmul(Aw, bestfit)
    return bestfit, cov, chi2, rchi2, fitfunc, resids


nplot = 200
zeroArr = array.array('d',[0]*nplot)
f0 = TF1("pol0","0",0,1e7)

linfits = defaultdict(list)
linpols = defaultdict(list)
linpolgraphs = defaultdict(list)
fitresults = defaultdict(list)
yErrArrs = defaultdict(list)
for iHdu, q in enumerate(hdus):
    for maxdeg in degrees:
        bestfit, cov, chi2, rchi2, fitfunc, resids = linFit(data[q], initialpars[q][1], maxdeg)
        print("chi2 %f, reduced chi2 %f, params %s" % (chi2, rchi2, str(bestfit)))
        fitresults[q].append((fitfunc,resids))

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

c.Clear()
c.Divide(1,4)
for iHdu, q in enumerate(hdus):
    iDeg = len(degrees)-1
    bestfit = linfits[q][iDeg]
    fitfunc, resids = fitresults[q][iDeg]
    t_arr, r_arr, rErr_arr = data[q]
    zero_arr = array.array('d',[0]*len(t_arr))
    one_arr = array.array('d',[1]*len(t_arr))

    grdata = TGraphErrors(len(t_arr), t_arr, r_arr, zero_arr, rErr_arr)
    grfit = TGraph(len(t_arr), t_arr, array.array('d', fitfunc))
    objects.append(grdata)
    objects.append(grfit)

    c.cd(1 + iHdu*2)
    gPad.SetLogy(1)
    gPad.SetLogx(1)
    grdata.Draw("A*")
    grfit.Draw("C")
    grfit.SetLineColor(2)
    c.cd(2 + iHdu*2)
    gPad.SetLogx(1)
    grresid = TGraphErrors(len(t_arr), t_arr, array.array('d', resids), zero_arr, one_arr)
    objects.append(grresid)
    grresid.Draw("A*")
    #grresid.GetYaxis().SetRangeUser(-1,1)
    f0.SetLineColor(2)
    #f0_clone = f0.Clone()
    #objects.append(f0_clone)
    f0.DrawCopy("lsame")
c.cd()
c.Print(outfilename+".pdf")



c.Print(outfilename+".pdf]")
outfile.Write()
outfile.Close()

