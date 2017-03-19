'''''''''''''''''''''''''''''''''
@Author : Vic P.
@Email  : vic4key@gmail.com
@Name   : CNNs Sample
'''''''''''''''''''''''''''''''''

import sys 
import vp as rg

LINE_FIXED = 50

Target = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, +1, -1, -1, -1, -1, -1, +1, -1],
    [-1, -1, +1, -1, -1, -1, +1, -1, -1],
    [-1, -1, -1, +1, -1, +1, -1, -1, -1],
    [-1, -1, -1, -1, +1, -1, -1, -1, -1],
    [-1, -1, -1, +1, -1, +1, -1, -1, -1],
    [-1, -1, +1, -1, -1, -1, +1, -1, -1],
    [-1, +1, -1, -1, -1, -1, -1, +1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1]
]

Features = [
    [
        [+1, -1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
    ],
    [
        [+1, -1, +1],
        [-1, +1, -1],
        [+1, -1, +1]
    ],
    [
        [-1, -1, +1],
        [-1, +1, -1],
        [+1, -1, -1]
    ]
]

def Title(text, allow = True):
    if not allow: return
    print text.center(LINE_FIXED, "-")
    return

def Print(target, allow = True):
    if not allow: return
    if target == None: return None
    m, n = len(target), len(target[0])
    if m == 0 or n == 0: return None

    for row in target:
        for v in row:
            if v >= 0: print "+%0.2f " % v,
            else: print "%0.2f " % v,
        print ""

    return

def Split(target, M, N, X, Y, Force = False):
    if target == None: return None
    m, n = len(target), len(target[0])
    if m == 0 or n == 0: return None

    if M > m or N > n: return None
    if X > m or Y > n: return None

    if not Force:
        if X + M > m or Y + N > n: return None

    l = []
    for MX, row in enumerate(target):
        if not Force:
            if X <= MX < X + M: l.append(row[Y:Y+N])
        else:
            if X <= MX < X + M:
                v = []
                if n - Y < N : v = row[Y:]
                else: v = row[Y:Y+N]
                l.append(v)

    return l

def CalcConv(target, feature):
    if target == None: return None
    if feature == None: return None

    Tm, Tn = len(target), len(target[0])
    Fm, Fn = len(feature), len(feature[0])
    if Tm == 0 or Tn == 0: return None
    if Fm == 0 or Fn == 0: return None

    if Tm != Tn: return None
    if Tm != Fm or Tn != Fn: return None

    ltarget = []
    for row in target: ltarget.extend(row)

    lfeature = []
    for row in feature: lfeature.extend(row)

    T, z = 0.0, Fm * Fn
    for i in xrange(0, z): T += ltarget[i] * lfeature[i]

    return round(T / z, 2)

def Convolution(target, feature):
    if target == None: return None
    if feature == None: return None
    Tm, Tn = len(target), len(target[0])
    Fm, Fn = len(feature), len(feature[0])
    if Tm == 0 or Tn == 0: return None
    if Fm == 0 or Fn == 0: return None

    l = []
    for X in xrange(0, Tm - Fm + 1):
        column = []
        for Y in xrange(0, Tn - Fn + 1):
            part = Split(target, Fm, Fn, X, Y)
            weight = CalcConv(part, feature)
            column.append(weight)
        l.append(column)

    return l

def ReLU(target, threshold = 0.0, value = 0.0):
    if target == None: return None
    return map(lambda row: map(lambda e: value if e < threshold else e, row), target)

def Pooling(target, M, N):
    if target == None: return None
    m, n = len(target), len(target[0])
    if m == 0 or n == 0: return None

    if M > m or N > n: return None

    l = []
    for X in xrange(0, m, M):
        _row = []
        for Y in xrange(0, n, N):
            part = Split(target, M, N, X, Y, True)
            lpart = []
            for row in part: lpart.extend(row)
            _row.append(max(lpart))
        l.append(_row)

    return l

def Layer(target, feature, pool = [2, 2], debug = True):
    result = Target

    Title("Convolution", debug)
    result = Convolution(result, feature)
    Print(result, debug)

    Title("ReLU", debug)
    result = ReLU(result)
    Print(result, debug)

    Title("Pooling", debug)
    result = Pooling(result, pool[0], pool[1])
    Print(result, debug)

    Title("Pooling", debug)
    result = Pooling(result, pool[0], pool[1])
    Print(result, debug)

    return result

def main():
    try:
        print "Howdy, Vic P."
        for i, Feature in enumerate(Features):
            print ("Feature[%d]" % i).center(LINE_FIXED, "*")
            result = Layer(Target, Feature, debug=False)
            Print(result)
    except (Exception, KeyboardInterrupt): rg.LogException(sys.exc_info())

if __name__ == "__main__":
    main()
    sys.exit(0)
