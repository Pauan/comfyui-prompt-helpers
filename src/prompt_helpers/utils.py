def clamp(value, low, high):
    return max(low, min(value, high))


def fold(list, f):
    seen = False
    output = None

    for item in list:
        if seen:
            output = f(output, item)

        else:
            seen = True
            output = item

    if seen:
        return output
    else:
        return None
