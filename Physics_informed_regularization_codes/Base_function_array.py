
import numpy
from random import *
import math

def director(A, B):
    return (A[1] - B[1]) / (sA[0] - B[0])


def coord_origin(A, B):
    return A[1] - director(A, B) * A[0]


def vec(A, B):
    return [(B[0] - A[0]), (B[1] - A[1])]


def vect_ortho(A, B):
    def y(x): return (((B[0] - A[0]) * (B[0] - x)) / (B[1] - A[1]) + B[1])
    return y


def norm(A, B):
    return (((A[0] - B[0])**2 + (A[1] - B[1])**2)**(1 / 2))


def center(A, B):
    return Point((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)


def scalaire(D, A, C, B):
    return (A[0] - D[0]) * (B[0] - C[0]) + (A[1] - D[1]) * (B[1] - C[1])


def x_ortho_L(X0, X1, L):
    return ((X0[0]) * (norm(X0, X1)**2) - L * ((X0[1] - X1[1])) * (norm(X0, X1))) / (norm(X0, X1)**2)


def x_ortho_Lp(X0, X1, L):
    return ((X0[0]) * (norm(X0, X1)**2) + L * (X0[1] - X1[1]) * (norm(X0, X1))) / (norm(X0, X1)**2)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Return true if line segments AB and CD intersect

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def angle_(A, G, B):
    dot = (A[0] - G[0]) * (B[0] - G[0]) + (A[1] - G[1]) * \
        (B[1] - G[1])      # dot product
    det = (A[0] - G[0]) * (B[1] - G[1]) - (A[1] - G[1]) * \
        (B[0] - G[0])      # determinant
    angle = math.atan2(det, dot)
    return angle

def diff_angle(A, G, B, Gp):
    dot = (A[0] - G[0]) * ((B[0] - (Gp[0] - G[0])) - G[0]) + (A[1] -
                                                              G[1]) * ((B[1] - (Gp[1] - G[1])) - G[1])      # dot product
    det = (A[0] - G[0]) * ((B[1] - (Gp[1] - G[1])) - G[1]) - (A[1] -
                                                              G[1]) * ((B[0] - (Gp[0] - G[0])) - G[0])      # determinant
    angle = math.atan2(det, dot)
    return angle

def Angle(A, G, B, Gp):
    angle = torch.atan2(A[1] - G[1], A[0] - G[0]) - \
        torch.atan2(B[1] - Gp[1], B[0] - Gp[0])
    return angle

def Surface_larva(Cont):
    S = 0
    for n in range(0, int(len(Cont)) - 1):
        n = n % len(Cont)
        if n != int(len(Cont) / 2):
            S += abs(0.5 * numpy.cross([Cont[int(n), :] - Cont[int(-n - 1), :]],
                                       [Cont[int(n), :] - Cont[int(n + 1), :]]))
    return S
