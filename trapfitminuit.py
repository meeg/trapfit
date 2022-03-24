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

os.system("gcc -shared -fPIC -o trapfit.so trapfit.c")

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

outfilename = "trapfitminuit_"+splitext(basename(filename))[0]
outfile = TFile(outfilename+".root","recreate")
gStyle.SetOptStat(0)
gStyle.SetOptFit(1)
c = TCanvas("c","c",1200,900);
c.Print(outfilename+".pdf[")
outfile.cd()

zero = TF1("zero","0",-1e7,1e7)

#dDC/dE(t) = 86400 * initial density * decay rate * surviving fraction(t)
#surviving fraction = exp(-t*decay rate)
#decay rate = C * T^2 * X * sigma * exp(-E/kT)
#(janesick eq. 8.29, 8.32, example 8.28)
#C = 1.6e21 cm^-2 sec^-1 K^-2
#assume X*sigma = 1e-15 cm^-2
#k = 8.62e-5 ev/K

#def decayrate(E):
#    temp = 170 #K
#    kt = 8.62e-5*temp
#    return 1.6e21 * 1e-15 * np.power(temp,2) * np.exp(-E/kt)
#def dc_fitfunc(x, p):
#    dc_eq = p[0]
#
#    density = Polynomial((p[2], p[3]))
#    def dc_integrand(E,t):
#        return density(E) * decayrate(E) * np.exp(-t*decayrate(E))
#    y,err = integrate.quad(dc_integrand, 0.0, 1.0, args=(x[0] - dt))
#
#    return 86400*y + dc_eq

Math.MinimizerOptions.SetDefaultMinimizer("Minuit","Simplex")

lib = ctypes.CDLL(os.path.abspath('trapfit.so'))

lib.dc_integrand.restype = ctypes.c_double
lib.dc_integrand.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))
func = LowLevelCallable(lib.dc_integrand)

def dc_polX(degree):
    def dc_fitfunc(x, p):
        #x = time
        p_arr = np.frombuffer(p, dtype=np.float64, count=degree+3)
        dc_eq = p[0]

        y,err = integrate.quad(func, 0.0, 1.0, args=(x[0], *p_arr[1:]))
        return y + dc_eq
    return dc_fitfunc


objects = [] #we don't use this, but it prevents the histograms from getting garbage-collected
def doFits(dataTuple, fitfunc, name):
    t_arr, r_arr, rErr_arr = dataTuple
    g = TGraphErrors(len(t_arr), t_arr, r_arr, 0, rErr_arr)
    #fitfunc.SetParameter(0,t[100]*r[100])
    #s = g.Fit(fitfunc,"SM")
    s = g.Fit(fitfunc,"S")
    pars = array.array('d', np.frombuffer(s.GetParams(), dtype=np.float64, count=fitfunc.GetNpar()))
    print(pars)
    cov = s.GetCovarianceMatrix()
    cov.Write(name+"_cov")
    g.SetTitle(name)
    g.GetXaxis().SetTitle("time elapsed [s]")
    g.GetYaxis().SetTitle("DC [e-/pix/day]")
    g.Write(name)
    res_arr = array.array('d')
    resErr_arr = array.array('d')
    for i, time in enumerate(t_arr):
        res_arr.append((r_arr[i] - fitfunc.Eval(time))/rErr_arr[i])
        resErr_arr.append(1.0)
    
    gRes = TGraphErrors(len(t_arr), t_arr, res_arr, 0, resErr_arr)
    gRes.SetTitle(name+"_res")
    gRes.Write(name+"_res")
    gRes.GetXaxis().SetTitle("time elapsed [s]")
    gRes.GetYaxis().SetTitle("normalized residual")
    objects.append(g)
    objects.append(gRes)
    return g, gRes, pars

def testFunc(fitfunc, name, oldpars):
    c.Clear()
    c.Divide(1,4)
    newpars = {}

    for iHdu, q in enumerate(hdus):
        print(oldpars[q])
        fitfunc.SetParameters(oldpars[q])
        #fitfunc.FixParameter(0,oldpars[q][0])
        #fitfunc.FixParameter(1,oldpars[q][1])
        g, gRes, pars = doFits(data[q], fitfunc, "q{0}_{1}".format(q,name))
        newpars[q] = pars
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
    return newpars

initialpars = {}
initialpars[2] = array.array('d', [0.58, 4.37e2, 2.51e3, -2.95e3])
initialpars[3] = array.array('d', [0.66, 3.51e2, 2.65e3, -3.18e3])

pols = defaultdict(list)

for degree in range(1,4):
    npars = degree+3
    polname = "dc_pol%d"%(degree)
    py_pol = dc_polX(degree)
    dc_pol = TF1(polname, py_pol, 0.0, 1e7, npars)
    dc_pol.SetParName(0, "dc_eq")
    dc_pol.SetParName(1, "t_offset")
    dc_pol.SetNpx(5000)

    newresults = testFunc(dc_pol, polname, initialpars)
    print(newresults)
    for q in hdus:
        newpol = TF1("pol%d_%d"%(degree,q),"pol%d(0)"%(degree),0,1.0)
        newpol.SetParameters(array.array('d',newresults[q][2:]))
        pols[q].append(newpol)
        newpars = array.array('d', np.zeros(npars+1))
        for i in range(npars):
            newpars[i] = newresults[q][i]
        initialpars[q] = newpars
        #print(newpars)


for q in hdus:
    c.Clear()
    c.SetLogx(0)
    c.SetLogy(0)
    for iPol, pol in enumerate(pols[q]):
        if iPol==0:
            pol.Draw()
        else:
            pol.Draw("same")
        pol.SetLineColor(iPol+2)
    c.Print(outfilename+".pdf")
#c.SetLogy()
#c.SetLogx()
#t, r, rErr = data[2]
#g = TGraphErrors(len(t), t, r, 0, rErr)
#g.Draw("A*")
##dc_tf1.Draw("same")
#g.Fit(dc_pol1)


c.Print(outfilename+".pdf]")
outfile.Write()
outfile.Close()
