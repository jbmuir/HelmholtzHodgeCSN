using ProgressMeter
using .Threads
using Lasso
using NPZ
using Statistics
using PyCall
using JLD
using SparseArrays

py"""
import scipy.sparse as sp

def spload(x):
    return sp.load_npz(x)

def spsave(x):
    return sp.save_npz("Scratch/results.npz", x)
"""

PyObject(S::SparseMatrixCSC) =
    pyimport("scipy.sparse").csc_matrix((S.nzval, S.rowval .- 1, S.colptr .- 1), shape=size(S))

G = npzread("Scratch/G.npy")

struct InvResult{T,S}
    i::Int
    λmin::T
    dn::T
    m::S
end

function fit_model(G,dv,i)
    dn = std(dv)
    dv ./= dn
    path = fit(LassoPath, G, dv, λminratio = 1e-3, α=0.95, standardize=false, intercept=false, maxncoef=size(G,2)) 
    λmin = path.λ[argmin(aicc(path))]
    m = coef(path; select=MinAICc())
    return InvResult(i, λmin, dn, m)
end

d = npzread("Scratch/d.npy")

p = Progress(size(d,2), dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

results = Array{Any,1}(undef, size(d,2))

@threads for i = 1:size(d,2)
    results[i] = fit_model(G,d[:,i],i)
    next!(p)
end

sres = reduce(hcat, [x.m.*x.dn for x in results])
py"spsave"(PyObject(sres))
