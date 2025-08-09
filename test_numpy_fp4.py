import subprocess
import time

print("Testing NumPy FP4 implementation...")
start = time.time()
result = subprocess.run(['python', 'poly_fp4_numpy.py'], capture_output=True, text=True)
end = time.time()

print(f"Output: {result.stdout}")
if result.stderr:
    print(f"Errors: {result.stderr}")
print(f"Runtime: {end - start:.2f}s")