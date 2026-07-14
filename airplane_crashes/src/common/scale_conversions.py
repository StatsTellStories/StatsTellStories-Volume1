def format_number(n):
    abs_n = abs(n)

    if abs_n >= 1e9:
        return f"{n / 1e9:.2f}".rstrip("0").rstrip(".") + " B"
    elif abs_n >= 1e6:
        return f"{n / 1e6:.2f}".rstrip("0").rstrip(".") + " M"
    elif abs_n >= 1e4:
        return f"{n / 1e3:.1f}".rstrip("0").rstrip(".") + " k"
    else:
        return str(n)