using Pkg

Pkg.activate(".")
Pkg.instantiate()
#
using MultivariateStats
#using PyCall
#
#
##import matplotlib
#pyimport("matplotlib")

# define an array 
x = rand(10, 10)
model_pca = length(x)
