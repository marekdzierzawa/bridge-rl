"""Phase 1: the tanh reward compared with a reward in IMPs."""
import numpy as np

from wspolne import Report

IMP_THRESHOLDS = (20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750,
                  900, 1100, 1300, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000)
DEVIATIONS = (50, 100, 200, 300, 500, 800, 1100, 1400, 2000)


def imp(points):
    return sum(1 for p in IMP_THRESHOLDS if abs(points) >= p)


def main():
    P = Report("f1_imp")
    P("%12s %14s %8s %14s" % ("odchylenie", "tanh(x/1000)", "IMP", "IMP / 24"))
    for x in DEVIATIONS:
        P("%12d %14.3f %8d %14.3f" % (x, np.tanh(x / 1000), imp(x), imp(x) / 24))
    P("")
    for big, small in [(1400, 100), (1400, 300), (800, 100)]:
        P("%5d wobec %4d: tanh %5.1f razy, IMP %5.1f razy"
          % (big, small, np.tanh(big / 1000) / np.tanh(small / 1000), imp(big) / imp(small)))
    P.close()


if __name__ == "__main__":
    main()
