#!/usr/bin/env python
#plot confidence bands for a pol2 fit
#based on https://root.cern/doc/master/ConfidenceIntervals_8C.html

import array
import numpy as np
from ROOT import gROOT, gStyle, TCanvas, TGraphErrors, TVirtualFitter, TVectorD

gROOT.SetBatch(True)
gStyle.SetOptFit(1)

outfilename = "conf_demo"
c = TCanvas("c","c",1200,900)
c.Print(outfilename+".pdf[")

data = []
ndata = 15
for x in np.linspace(1,3,ndata):
    data.append([x,0,3-x,0.1])
xArr, xErrArr, yArr, yErrArr = [array.array('d',x) for x in zip(*data)]
graph = TGraphErrors(len(xArr),xArr,yArr,xErrArr,yErrArr)
#graph.Draw("AP")

nplot = 100
xArr = array.array('d',np.linspace(0,4,nplot))
xErrArr = array.array('d',[0]*nplot)
yArr = array.array('d',[0]*nplot)
yErrArr = array.array('d',[0]*nplot)
funcs = ["pol"+str(dim) for dim in range(1,5)]
funcs.append("expo")
for func in funcs:
    s = graph.Fit(func, "S")

    grint = TGraphErrors(nplot, xArr, yArr)
    grint.SetTitle(func)
    TVirtualFitter.GetFitter().GetConfidenceIntervals(grint,0.68) #1-sigma intervals
    grint.SetFillColor(2)
    grint.SetFillStyle(3005)
    grint.Draw("A4")

    graph.Draw("P")

    s.GetCovarianceMatrix().Print()

    f = graph.GetFunction(func)
    #f.Print()
    npar = f.GetNpar()
    coeffVec = TVectorD(npar) # derivative of f(x) wrt the i'th parameter
    for iX, x in enumerate(xArr):
        yArr[iX] = f.Eval(x)
        if func[:3]=="pol":
            for i in range(npar):
                coeffVec[i] = x**i # for a polynomial, df(x)/dp_i is just x^i
        else: #expo: y = A*exp(b*x)
            coeffVec[0] = yArr[iX]/f.GetParameter(0) # dy/dA = y/A
            coeffVec[1] = x*yArr[iX] # dy/db = x*y
        yErrArr[iX] = np.sqrt(s.GetCovarianceMatrix().Similarity(coeffVec)) # error propagation: multiply the covariance matrix for p_i by df/dp to get the variance of f
        #print(x, yArr[iX], yErrArr[iX])
    grcov = TGraphErrors(nplot, xArr, yArr, xErrArr, yErrArr)
    grcov.SetFillColor(4)
    grcov.SetFillStyle(3004)
    grcov.Draw("4")

    #if func=="expo":
    #    c.SetLogy(1)
    #else:
    #    c.SetLogy(0)

    c.Print(outfilename+".pdf")


c.Print(outfilename+".pdf]")
