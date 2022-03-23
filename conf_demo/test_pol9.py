#!/usr/bin/env python
#plot confidence bands for a pol2 fit
#based on https://root.cern/doc/master/ConfidenceIntervals_8C.html

import sys
import array
import numpy as np
from ROOT import gROOT, gStyle, TCanvas, TGraphErrors, TVirtualFitter, TVectorD, TF1, TMatrixDSym

gROOT.SetBatch(True)
gStyle.SetOptFit(1)

outfilename = "test_pol9"
c = TCanvas("c","c",1200,900)
c.Print(outfilename+".pdf[")


pars3 = [-32866.11886109412, 178012.35741239786, -305899.5805425644, 171342.0753324926]
cov3 = [[  50.80879923, -42.33039705, -53.65272483, -57.75137626],
        [ -42.33039705, 189.31440076,-107.98336772,-167.51044866],
        [ -53.65272483,-107.98336772, 594.44316737,-407.60458608],
        [ -57.75137626,-167.51044866,-407.60458608,1595.04804139]]

pars9 = [-33466.38381379843, 171772.06927740574, -303099.2961010784, 210994.07364477217, 32627.02584911883, -32539.423317030072, -200206.95852068067, -254305.08667692542, 415543.39513365924, 254982.1718918383]
cov9 = [[ 2.50135243e+01, -1.56053674e+01, -2.25444424e+01, -3.09023622e+01,
        -3.91365181e+01, -4.30594642e+01, -3.31318711e+01,  1.04240297e+01,
         1.26321735e+02,  3.87356398e+02],
       [-1.56053674e+01,  8.81192180e+01, -3.55861152e+01, -5.08337930e+01,
        -6.87567759e+01, -8.55953749e+01, -9.15313411e+01, -6.47542835e+01,
         3.92741339e+01,  3.06765184e+02],
       [-2.25444424e+01, -3.55861152e+01,  2.96223638e+02, -8.23585660e+01,
        -1.19543341e+02, -1.66225247e+02, -2.17581225e+02, -2.58583453e+02,
        -2.53458107e+02, -1.26490227e+02],
       [-3.09023622e+01, -5.08337930e+01, -8.23585660e+01,  9.57818189e+02,
        -2.05710049e+02, -3.16303607e+02, -4.76244645e+02, -6.99920133e+02,
        -9.99218898e+02, -1.37497543e+03],
       [-3.91365181e+01, -6.87567759e+01, -1.19543341e+02, -2.05710049e+02,
         2.99485586e+03, -5.91158195e+02, -9.87726540e+02, -1.63497102e+03,
        -2.68182814e+03, -4.36005330e+03],
       [-4.30594642e+01, -8.55953749e+01, -1.66225247e+02, -3.16303607e+02,
        -5.91158195e+02,  9.09069167e+03, -1.96993184e+03, -3.52164654e+03,
        -6.21752195e+03, -1.08505279e+04],
       [-3.31318711e+01, -9.15313411e+01, -2.17581225e+02, -4.76244645e+02,
        -9.87726540e+02, -1.96993184e+03,  2.68670737e+04, -7.19372705e+03,
        -1.32990691e+04, -2.41496581e+04],
       [ 1.04240297e+01, -6.47542835e+01, -2.58583453e+02, -6.99920133e+02,
        -1.63497102e+03, -3.52164654e+03, -7.19372705e+03,  7.74971121e+04,
        -2.69893836e+04, -5.02870713e+04],
       [ 1.26321735e+02,  3.92741339e+01, -2.53458107e+02, -9.99218898e+02,
        -2.68182814e+03, -6.21752195e+03, -1.32990691e+04, -2.69893836e+04,
         2.18613024e+05, -1.00061052e+05],
       [ 3.87356398e+02,  3.06765184e+02, -1.26490227e+02, -1.37497543e+03,
        -4.36005330e+03, -1.08505279e+04, -2.41496581e+04, -5.02870713e+04,
        -1.00061052e+05,  6.04178009e+05]]

pars9d = list(pars9)
for i,x in enumerate(pars3):
    pars9d[i] -= x

#scaleCov = 100.0
scaleCov = 1.0

covmat3 = TMatrixDSym(4)
for i in range(4):
    for j in range(4):
        covmat3[i][j] = cov3[i][j]*scaleCov
covmat3.Print()

covmat9 = TMatrixDSym(10)
for i in range(10):
    for j in range(10):
        covmat9[i][j] = cov9[i][j]*scaleCov
covmat9.Print()

f3 = TF1("pol3","pol3(0)",0,1.0)
f3.SetParameters(array.array('d',pars3))
f9 = TF1("pol9","pol9(0)",0,1.0)
f9.SetParameters(array.array('d',pars9))
f3d = TF1("pol0","0",0,1.0)
f9d = TF1("pol9d","pol9(0)",0,1.0)
f9d.SetParameters(array.array('d',pars9d))

nplot = 200
xArr = array.array('d',np.linspace(-2.0,1.0,nplot))
zeroArr = array.array('d',[0]*nplot)

yArr3 = array.array('d',[0]*nplot)
yErrArr3 = array.array('d',[0]*nplot)
npar3 = f3.GetNpar()
coeffVec3 = TVectorD(npar3) # derivative of f(x) wrt the i'th parameter

yArr9 = array.array('d',[0]*nplot)
yErrArr9 = array.array('d',[0]*nplot)
npar9 = f9.GetNpar()
coeffVec9 = TVectorD(npar9) # derivative of f(x) wrt the i'th parameter

yArr9d = array.array('d',[0]*nplot)

for iX, x in enumerate(xArr):
    yArr3[iX] = f3.Eval(x)
    yArr9[iX] = f9.Eval(x)
    yArr9d[iX] = f9d.Eval(x)
    for i in range(npar3):
        coeffVec3[i] = x**i # for a polynomial, df(x)/dp_i is just x^i
    yErrArr3[iX] = np.sqrt(covmat3.Similarity(coeffVec3)) # error propagation: multiply the covariance matrix for p_i by df/dp to get the variance of f
    for i in range(npar9):
        coeffVec9[i] = x**i # for a polynomial, df(x)/dp_i is just x^i
    yErrArr9[iX] = np.sqrt(covmat9.Similarity(coeffVec9)) # error propagation: multiply the covariance matrix for p_i by df/dp to get the variance of f
    #print(x, yArr[iX], yErrArr[iX])
#print(yErrArr)
grcov3 = TGraphErrors(nplot, xArr, yArr3, zeroArr, yErrArr3)
grcov3.SetFillColor(2)
grcov3.SetFillStyle(3006)
grcov3.GetXaxis().SetRangeUser(0.45,0.65)
grcov3.GetYaxis().SetRangeUser(600,1200)
grcov3.Draw("A3")
grcov3.SetTitle("pol3 and pol9 fits")
f3.Draw("lsame")
f3.SetLineColor(2)
grcov9 = TGraphErrors(nplot, xArr, yArr9, zeroArr, yErrArr9)
grcov9.SetFillColor(4)
grcov9.SetFillStyle(3007)
grcov9.Draw("3")
f9.Draw("lsame")
f9.SetLineColor(4)
c.Print(outfilename+".pdf")

grcov3d = TGraphErrors(nplot, xArr, zeroArr, zeroArr, yErrArr3)
grcov3d.SetFillColor(2)
grcov3d.SetFillStyle(3005)
grcov3d.GetXaxis().SetRangeUser(0.3,0.8)
grcov3d.GetYaxis().SetRangeUser(-20,20)
grcov3d.Draw("A3")
grcov3d.SetTitle("deviations from pol3 fit")
f3d.Draw("lsame")
f3d.SetLineColor(2)
grcov9d = TGraphErrors(nplot, xArr, yArr9d, zeroArr, yErrArr9)
grcov9d.SetFillColor(4)
grcov9d.SetFillStyle(3004)
grcov9d.Draw("3")
f9d.Draw("lsame")
f9d.SetLineColor(4)
c.Print(outfilename+".pdf")

grcov9z = TGraphErrors(nplot, xArr, zeroArr, zeroArr, yErrArr9)
grcov9z.SetFillColor(4)
grcov9z.SetFillStyle(3004)
grcov9z.Draw("A3")
grcov9z.SetTitle("confidence band of pol9 fit")
grcov9z.GetXaxis().SetRangeUser(0.3,0.8)
#grcov9z.GetXaxis().SetRangeUser(-1.0,0.8)
grcov9z.GetYaxis().SetRangeUser(-20,20)
f3d.Draw("lsame")
f3d.SetLineColor(4)
c.Print(outfilename+".pdf")

c.Print(outfilename+".pdf]")

