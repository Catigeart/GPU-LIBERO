"""Private FP64 Horner sin45/cos44 for |half-angle| <= 4.
Each decimal literal round trips to the shown exact binary64 hexadecimal value.
"""
import libero.gpu.precision as scalar
import warp as wp
wp.set_module_options({'enable_backward':False,'fuse_fp':True,'fast_math':False})

@wp.func
def sin45(z: scalar.wp_real) -> scalar.wp_real:
    t = z * z
    s = scalar.wp_real(8.359650847182804e-57)  # 0x1.a3cb872220648p-187
    s = s * t + scalar.wp_real(-1.6552108677421951e-53)  # -0x1.95db45257e512p-176
    s = s * t + scalar.wp_real(2.9893108271424046e-50)  # 0x1.65e61c39d0241p-165
    s = s * t + scalar.wp_real(-4.902469756513544e-47)  # -0x1.1e99449a4bacep-154
    s = s * t + scalar.wp_real(7.265460179153071e-44)  # 0x1.9ec8d1c94e85bp-144
    s = s * t + scalar.wp_real(-9.67759295863189e-41)  # -0x1.0dc59c716d91fp-133
    s = s * t + scalar.wp_real(1.151633562077195e-37)  # 0x1.3981254dd0d52p-123
    s = s * t + scalar.wp_real(-1.216125041553518e-34)  # -0x1.434d2e783f5bcp-113
    s = s * t + scalar.wp_real(1.1309962886447716e-31)  # 0x1.259f98b4358adp-103
    s = s * t + scalar.wp_real(-9.183689863795546e-29)  # -0x1.d1ab1c2dccea3p-94
    s = s * t + scalar.wp_real(6.446950284384474e-26)  # 0x1.3f3ccdd165fa9p-84
    s = s * t + scalar.wp_real(-3.868170170630684e-23)  # -0x1.761b41316381ap-75
    s = s * t + scalar.wp_real(1.9572941063391263e-20)  # 0x1.71b8ef6dcf572p-66
    s = s * t + scalar.wp_real(-8.22063524662433e-18)  # -0x1.2f49b46814157p-57
    s = s * t + scalar.wp_real(2.8114572543455206e-15)  # 0x1.952c77030ad4ap-49
    s = s * t + scalar.wp_real(-7.647163731819816e-13)  # -0x1.ae7f3e733b81fp-41
    s = s * t + scalar.wp_real(1.6059043836821613e-10)  # 0x1.6124613a86d09p-33
    s = s * t + scalar.wp_real(-2.505210838544172e-08)  # -0x1.ae64567f544e4p-26
    s = s * t + scalar.wp_real(2.7557319223985893e-06)  # 0x1.71de3a556c734p-19
    s = s * t + scalar.wp_real(-0.0001984126984126984)  # -0x1.a01a01a01a01ap-13
    s = s * t + scalar.wp_real(0.008333333333333333)  # 0x1.1111111111111p-7
    s = s * t + scalar.wp_real(-0.16666666666666666)  # -0x1.5555555555555p-3
    s = s * t + scalar.wp_real(1.0)  # 0x1.0000000000000p+0
    return z * s

@wp.func
def cos44(z: scalar.wp_real) -> scalar.wp_real:
    t = z * z
    c = scalar.wp_real(3.7618428812322616e-55)  # 0x1.272b1b03fec6ap-181
    c = c * t + scalar.wp_real(-7.117406731291439e-52)  # -0x1.10af527530de8p-170
    c = c * t + scalar.wp_real(1.2256174391283858e-48)  # 0x1.ca8ed42a12ae3p-160
    c = c * t + scalar.wp_real(-1.911963205040282e-45)  # -0x1.5d4acb9c0c3abp-149
    c = c * t + scalar.wp_real(2.6882202662866363e-42)  # 0x1.df983290c2ca9p-139
    c = c * t + scalar.wp_real(-3.387157535521162e-39)  # -0x1.2710231c0fd7ap-128
    c = c * t + scalar.wp_real(3.8003907548547434e-36)  # 0x1.434d2e783f5bcp-118
    c = c * t + scalar.wp_real(-3.7699876288159054e-33)  # -0x1.3932c5047d60ep-108
    c = c * t + scalar.wp_real(3.279889237069838e-30)  # 0x1.0a18a2635085dp-98
    c = c * t + scalar.wp_real(-2.4795962632247976e-27)  # -0x1.88e85fc6a4e5ap-89
    c = c * t + scalar.wp_real(1.6117375710961184e-24)  # 0x1.f2cf01972f578p-80
    c = c * t + scalar.wp_real(-8.896791392450574e-22)  # -0x1.0ce396db7f853p-70
    c = c * t + scalar.wp_real(4.110317623312165e-19)  # 0x1.e542ba4020225p-62
    c = c * t + scalar.wp_real(-1.5619206968586225e-16)  # -0x1.6827863b97d97p-53
    c = c * t + scalar.wp_real(4.779477332387385e-14)  # 0x1.ae7f3e733b81fp-45
    c = c * t + scalar.wp_real(-1.1470745597729725e-11)  # -0x1.93974a8c07c9dp-37
    c = c * t + scalar.wp_real(2.08767569878681e-09)  # 0x1.1eed8eff8d898p-29
    c = c * t + scalar.wp_real(-2.755731922398589e-07)  # -0x1.27e4fb7789f5cp-22
    c = c * t + scalar.wp_real(2.48015873015873e-05)  # 0x1.a01a01a01a01ap-16
    c = c * t + scalar.wp_real(-0.001388888888888889)  # -0x1.6c16c16c16c17p-10
    c = c * t + scalar.wp_real(0.041666666666666664)  # 0x1.5555555555555p-5
    c = c * t + scalar.wp_real(-0.5)  # -0x1.0000000000000p-1
    c = c * t + scalar.wp_real(1.0)  # 0x1.0000000000000p+0
    return c
