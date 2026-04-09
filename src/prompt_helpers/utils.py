def clamp(value, low, high):
    return max(low, min(value, high))


def snap_to_increment(value, increment):
    extra = value % increment

    if extra == 0:
        return value
    else:
        return value + (increment - extra)


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
