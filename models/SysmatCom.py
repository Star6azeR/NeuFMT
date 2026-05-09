def SysmatCom(S1_one, S2_one, S3_one, mua, mus, ref=1.4, alpha=2.7439):
    kap = 1 / (3 * (mua + mus))
    c0 = 0.3
    c = c0 / ref
    t1 = mua * c
    t2 = kap * c
    t3 = c / (2 * alpha)
    S = t1 * S1_one + t2 * S2_one + t3 * S3_one
    return S