from fractions import Fraction

def calculate_weights(inner_tardiness_ratio: float, lateness_ratio: float, max_denominator: int = 100):
    # 1) Tardiness/Earliness aufteilen
    tardiness_frac = Fraction(inner_tardiness_ratio).limit_denominator(max_denominator)
    tardiness = tardiness_frac.numerator
    earliness = tardiness_frac.denominator - tardiness

    # 2) Lateness/Deviation aufteilen
    lateness_frac = Fraction(lateness_ratio).limit_denominator(max_denominator)
    lateness_factor = lateness_frac.numerator
    dev_factor = lateness_frac.denominator - lateness_factor

    # 3) Summen bilden
    amount = tardiness + earliness

    # 4) Gewichte berechnen
    w_t = tardiness * lateness_factor
    w_e = earliness * lateness_factor
    w_dev = amount * dev_factor

    return w_t, w_e, w_dev


# Beispielaufruf
w_t, w_e, w_dev = calculate_weights(0.50, 0.666)
print(f"w_t = {w_t}, w_e = {w_e}, w_dev = {w_dev}")
