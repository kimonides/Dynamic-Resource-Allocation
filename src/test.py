def formatForCAT(ways, reverse=False):
        if reverse:
            pivot = 1 << 19
            res = 0
            for i in range(0,ways):
                print("ITERATION %s" % i )
                print("RES IS %s" % hex(res))
                print("PIVOT IS %s" % bin(pivot))
                res = res + pivot

                pivot = pivot >> 1
                
        else:
            res = 1 << ways - 1
            res = res + res - 1
        print(bin(res))
        return hex(res)

print(formatForCAT(4,True))