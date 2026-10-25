"""Host dimension checks for resident device buffers."""
INT32_MAX = 2**31-1

def checked_product(values, *, limit=INT32_MAX):
    result=1
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative Python integer dimensions required")
        result *= value
        if result > limit:
            raise OverflowError(f"index/byte product {result} exceeds {limit}")
    return result
