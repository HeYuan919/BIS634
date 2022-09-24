def temp_tester(normal_temp):
    def test_norm(real_temp):
        if abs(normal_temp - real_temp) <= 1:
            return True
        else:
            return False

    return test_norm


'''This is the Test code'''
human_tester = temp_tester(37)
chicken_tester = temp_tester(41.1)

print(chicken_tester(42))  # True -- i.e. not a fever for a chicken
print(human_tester(42))  # False -- this would be a severe fever for a human
print(chicken_tester(43))  # False
print(human_tester(35))  # False -- too low
print(human_tester(98.6))  # False -- normal in degrees F but our reference temp was in degrees C
