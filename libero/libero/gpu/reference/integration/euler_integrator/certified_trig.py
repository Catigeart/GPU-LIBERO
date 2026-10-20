"""Fixed binary64 Taylor device polynomials on |z| <= 4.

Coefficients are nearest binary64 to exact factorial rationals.
No GPU outputs enter their construction. See NUMERICS.md for bounds.
"""
import libero.gpu.precision as scalar

from fractions import Fraction
from math import factorial
import warp as wp

DOMAIN = 4.0
SIN_TAIL = Fraction(4**47, factorial(47))
COS_TAIL = Fraction(4**46, factorial(46))
SIN_COEFF = tuple(float(Fraction((-1)**k, factorial(2*k+1))) for k in range(23))
COS_COEFF = tuple(float(Fraction((-1)**k, factorial(2*k))) for k in range(23))

@wp.func
def sin45(z: scalar.wp_real) -> scalar.wp_real:
    zz = z*z
    p = scalar.wp_real(8.359650847182804e-57)
    p = p*zz + scalar.wp_real(-1.6552108677421951e-53)
    p = p*zz + scalar.wp_real(2.9893108271424046e-50)
    p = p*zz + scalar.wp_real(-4.902469756513544e-47)
    p = p*zz + scalar.wp_real(7.265460179153071e-44)
    p = p*zz + scalar.wp_real(-9.67759295863189e-41)
    p = p*zz + scalar.wp_real(1.151633562077195e-37)
    p = p*zz + scalar.wp_real(-1.216125041553518e-34)
    p = p*zz + scalar.wp_real(1.1309962886447716e-31)
    p = p*zz + scalar.wp_real(-9.183689863795546e-29)
    p = p*zz + scalar.wp_real(6.446950284384474e-26)
    p = p*zz + scalar.wp_real(-3.868170170630684e-23)
    p = p*zz + scalar.wp_real(1.9572941063391263e-20)
    p = p*zz + scalar.wp_real(-8.22063524662433e-18)
    p = p*zz + scalar.wp_real(2.8114572543455206e-15)
    p = p*zz + scalar.wp_real(-7.647163731819816e-13)
    p = p*zz + scalar.wp_real(1.6059043836821613e-10)
    p = p*zz + scalar.wp_real(-2.505210838544172e-08)
    p = p*zz + scalar.wp_real(2.7557319223985893e-06)
    p = p*zz + scalar.wp_real(-0.0001984126984126984)
    p = p*zz + scalar.wp_real(0.008333333333333333)
    p = p*zz + scalar.wp_real(-0.16666666666666666)
    p = p*zz + scalar.wp_real(1.0)
    return z*p

@wp.func
def cos44(z: scalar.wp_real) -> scalar.wp_real:
    zz = z*z
    p = scalar.wp_real(3.7618428812322616e-55)
    p = p*zz + scalar.wp_real(-7.117406731291439e-52)
    p = p*zz + scalar.wp_real(1.2256174391283858e-48)
    p = p*zz + scalar.wp_real(-1.911963205040282e-45)
    p = p*zz + scalar.wp_real(2.6882202662866363e-42)
    p = p*zz + scalar.wp_real(-3.387157535521162e-39)
    p = p*zz + scalar.wp_real(3.8003907548547434e-36)
    p = p*zz + scalar.wp_real(-3.7699876288159054e-33)
    p = p*zz + scalar.wp_real(3.279889237069838e-30)
    p = p*zz + scalar.wp_real(-2.4795962632247976e-27)
    p = p*zz + scalar.wp_real(1.6117375710961184e-24)
    p = p*zz + scalar.wp_real(-8.896791392450574e-22)
    p = p*zz + scalar.wp_real(4.110317623312165e-19)
    p = p*zz + scalar.wp_real(-1.5619206968586225e-16)
    p = p*zz + scalar.wp_real(4.779477332387385e-14)
    p = p*zz + scalar.wp_real(-1.1470745597729725e-11)
    p = p*zz + scalar.wp_real(2.08767569878681e-09)
    p = p*zz + scalar.wp_real(-2.755731922398589e-07)
    p = p*zz + scalar.wp_real(2.48015873015873e-05)
    p = p*zz + scalar.wp_real(-0.001388888888888889)
    p = p*zz + scalar.wp_real(0.041666666666666664)
    p = p*zz + scalar.wp_real(-0.5)
    p = p*zz + scalar.wp_real(1.0)
    return p
