import subprocess
import time

print("Testing fixed FP16...")
start = time.time()
result = subprocess.run(['python', 'poly_fp16.py'], capture_output=True, text=True)
print(f"FP16: {result.stdout.strip()} (Runtime: {time.time()-start:.2f}s)")
if result.stderr: print(f"Error: {result.stderr}")

print("\nTesting fixed FP8...")
start = time.time()
result = subprocess.run(['python', 'poly_fp8.py'], capture_output=True, text=True)
print(f"FP8: {result.stdout.strip()} (Runtime: {time.time()-start:.2f}s)")
if result.stderr: print(f"Error: {result.stderr}")