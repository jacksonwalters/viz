#the discrete log of an element x in a finite field F
def dlog(x, F):
    return x.log(F.multiplicative_generator()) if x != 0 else -1