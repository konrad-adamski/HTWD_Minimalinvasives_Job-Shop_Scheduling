from fractions import Fraction


tardiness_ration = 0.75
lateness_ratio = 0.75


tardiness_frac = Fraction(tardiness_ration)

tardiness = tardiness_frac.numerator
earliness = tardiness_frac.denominator - tardiness

print(f"w_t = {tardiness}, w_e = {earliness}")

print("-"*60)

lateness_frac = Fraction(lateness_ratio)

lateness_factor = lateness_frac.numerator
dev_factor = lateness_frac.denominator - lateness_factor

print(f"lateness_factor = {lateness_factor}, dev_factor = {dev_factor}")

print("-"*60)

amount = tardiness + earliness

w_t = tardiness * lateness_factor
w_e = earliness * lateness_factor

w_dev = amount * dev_factor


print(f"w_t = {w_t}, w_e = {w_e}, w_dev = {w_dev}")





