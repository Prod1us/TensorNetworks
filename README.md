#Tensor Networks

Implementations of 1D tensor networks algorithms in python.

## MPS decomposition 
Basic tensor network algorithm that decomposes an n-rank tensor into a mixed canonical matrix product state (MPS). 
In diagrammatic form, it looks like this

![d - bond dimensions](pictures/mps.svg)

This form of representing manybody states allows to perform 
useful calculations. For example we can easily calculate expectation values 
of one-site 

![](pictures/Ev1.svg)

or two-site local operators

![](pictures/Ev2.svg)


## Roadmap

Main goals for now (in priority order):


- [x] MPS decomposition algorithm 
- [ ] MPO Lanczos based algorithms 
- [ ] TDVP
