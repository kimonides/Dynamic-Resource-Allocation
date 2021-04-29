def addDictionaries(a, b):
    "merges b into a"
    res = {}
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                res[key] = addDictionaries(a[key], b[key])
            elif isinstance(a[key], dict) or isinstance(b[key], dict):
                raise Exception('Conflict')
            else:
                res[key] = float(a[key]) + float(b[key])
        else:
            res[key] = b[key]
    return res

def divideDictionaries(a, nr):
    res = {}
    for key in a:
        if isinstance(a[key], dict) :
            res[key] = divideDictionaries(a[key], nr)
        else:
            res[key] = float(a[key]) / nr
    return res