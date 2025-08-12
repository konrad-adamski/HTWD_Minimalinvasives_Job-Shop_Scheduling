from fractions import Fraction


inner_tardiness_ratio = 0.50
lateness_ratio = 0.75


tardiness_frac = Fraction(inner_tardiness_ratio).limit_denominator(100)

tardiness = tardiness_frac.numerator
earliness = tardiness_frac.denominator - tardiness

print(f"Tardiness ratio: {inner_tardiness_ratio},  {tardiness_frac = }")
print(f"w_t = {tardiness}, w_e = {earliness}")

print("-"*60)

lateness_frac = Fraction(lateness_ratio).limit_denominator(100)

lateness_factor = lateness_frac.numerator
dev_factor = lateness_frac.denominator - lateness_factor

print(f"lateness_factor = {lateness_factor}, dev_factor = {dev_factor}")

print("-"*60)

amount = tardiness + earliness

w_t = tardiness * lateness_factor
w_e = earliness * lateness_factor

w_dev = amount * dev_factor

print(f"w_t = {w_t}, w_e = {w_e}, w_dev = {w_dev}")

total_shift_number =25

for shift_number in range(1, total_shift_number + 1):
    print(f"shift_number = {shift_number}")



print("="*60)

