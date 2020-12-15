import logging
import numpy as np

def polynomial(coefficients, x_range=tuple(range(1000000))):
    result = [
            coefficients["a"] * (x ** 2) + coefficients["b"] * x + coefficients["c"]
            for x in x_range
        ]
    result = np.array(result)
    logging.info(f"Array size: {result.nbytes*10e-6}")
    return {
        "polynomial_output": result
    }
