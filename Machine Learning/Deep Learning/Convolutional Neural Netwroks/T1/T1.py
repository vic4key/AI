'''''''''''''''''''''''''''''''''
@Author : Vic P.
@Email  : vic4key@gmail.com
@Name   : CNNs Sample
'''''''''''''''''''''''''''''''''

import sys, cv2
import vp as rg
import numpy as np
import matplotlib.pyplot as plt

LINE_FIXED = 50

def I2B(Path, Fore = 1, Back = -1): # Convert an image to a binary matrix
    M = cv2.imread(Path, cv2.IMREAD_GRAYSCALE).astype(np.int8)
    M[M != -1] = Fore # Backgroud = White -> -1
    return M

Target = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, +1, -1, -1, -1, -1, -1, +1, -1],
    [-1, -1, +1, -1, -1, -1, +1, -1, -1],
    [-1, -1, -1, +1, -1, +1, -1, -1, -1],
    [-1, -1, -1, -1, +1, -1, -1, -1, -1],
    [-1, -1, -1, +1, -1, +1, -1, -1, -1],
    [-1, -1, +1, -1, -1, -1, +1, -1, -1],
    [-1, +1, -1, -1, -1, -1, -1, +1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1]
], np.int8)

Features = [
    np.array([
        [+1, -1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
    ], np.int8),
    np.array([
        [+1, -1, +1],
        [-1, +1, -1],
        [+1, -1, +1]
    ], np.int8),
    np.array([
        [-1, -1, +1],
        [-1, +1, -1],
        [+1, -1, -1]
    ], np.int8)
]

Target = I2B(r"./DataSet/Target.bmp")
Features = [
    I2B(r"./DataSet/Feature - 0.bmp"),
    I2B(r"./DataSet/Feature - 1.bmp"),
    I2B(r"./DataSet/Feature - 2.bmp"),
    I2B(r"./DataSet/Feature - 3.bmp"),
    I2B(r"./DataSet/Feature - 4.bmp"),
    I2B(r"./DataSet/Feature - 5.bmp"),
    I2B(r"./DataSet/Feature - 6.bmp"),
]

DEF_PLOT = 231

def Title(text, allow = True):
    if not allow: return
    print text.center(LINE_FIXED, "-")
    return

def Print(target, allow = True):
    if not allow: return
    if type(target) is not np.ndarray: return
    m, n = target.shape if len(target.shape) == 2 else (0, 0)
    if m == 0 or n == 0: return

    for row in target:
        for v in row:
            if v >= 0: print "+%0.2f " % v,
            else: print "%0.2f " % v,
        print ""

    return

def Plot(image, caption, N = 231, show = True):
    if not show: return
    plt.subplot(N), plt.imshow(image, "gray"), plt.title(caption)

def Split(target, M, N, X, Y, Force = False):
    if type(target) is not np.ndarray: return None
    m, n = target.shape if len(target.shape) == 2 else (0, 0)
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

    return np.array(l)

def Convolution(target, feature):
    if type(target) is not np.ndarray or type(feature) is not np.ndarray: return None
    Tm, Tn = target.shape if len(target.shape) == 2 else (0, 0)
    Fm, Fn = feature.shape if len(feature.shape) == 2 else (0, 0)
    if Tm == 0 or Tn == 0: return None
    if Fm == 0 or Fn == 0: return None

    l = []
    for X in xrange(0, Tm - Fm + 1):
        column = []
        for Y in xrange(0, Tn - Fn + 1):
            part = Split(target, Fm, Fn, X, Y)
            weight = round(np.sum(part * feature) / float(Fm * Fn), 2)
            column.append(weight)
        l.append(column)

    return np.array(l)

def ReLU(target, threshold = 0.0, value = 0.0):
    if type(target) is not np.ndarray: return None
    return np.array(map(lambda row: map(lambda e: value if e < threshold else e, row), target))

def Pooling(target, M, N):
    if type(target) is not np.ndarray: return None
    m, n = target.shape if len(target.shape) == 2 else (0, 0)
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

    return np.array(l)

def Evaluation(result, percent=50):
    if type(result) is not np.ndarray: return None
    Rmin, Rmax, Voted, N = [], [], 0, 0
    for row in result:
        Rmin.append(min(row))
        Rmax.append(max(row))
        Voted += len(filter(lambda v: v > float(percent / 100.0), row))
        N += len(row)
    return ([100 * min(Rmin), 100 * max(Rmax)], (Voted, N, float(100 * Voted / N)))

def Layer(target, feature, pool = [2, 2], loop = 1, msg = True, plot = True):
    result = Target

    for i in xrange(0, loop):
        nplot = DEF_PLOT

        caption = "Convolution"
        Title(caption, msg)
        result = Convolution(result, feature)
        Print(result, msg)
        Plot(result, caption, nplot, plot)
        nplot += 1

        caption = "ReLU"
        Title(caption, msg)
        result = ReLU(result)
        Print(result, msg)
        Plot(result, caption, nplot, plot)
        nplot += 1

        caption = "Pooling"
        Title(caption, msg)
        result = Pooling(result, pool[0], pool[1])
        Print(result, msg)
        Plot(result, caption, nplot, plot)
        nplot += 1

        caption = "Pooling"
        Title(caption, msg)
        result = Pooling(result, pool[0], pool[1])
        Print(result, msg)
        Plot(result, caption, nplot, plot)
        nplot += 1

        if plot: plt.show()

    return result

def main():
    try:
        print "Howdy, Vic P."
        for i, Feature in enumerate(Features):
            print ("Feature[%d]" % i).center(LINE_FIXED, "*")
            result = Layer(Target, Feature, msg = False, plot = True)
            eRange, eGuess = Evaluation(result, 68)
            print "[Min: %0.2d%%, Max: %0.2d%%] : [Voted: %0.2d, Total: %0.2d] -> Avg: %0.2d%%" % \
                (eRange[0], eRange[1], eGuess[0], eGuess[1], eGuess[2])
    except (Exception, KeyboardInterrupt): rg.LogException(sys.exc_info())

if __name__ == "__main__":
    main()
    sys.exit(0)
