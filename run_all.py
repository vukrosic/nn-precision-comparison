import subprocess
import time

precisions = ['fp16', 'bf16', 'fp8', 'fp4']

for precision in precisions:
    print(f"\n{'='*50}")
    print(f"Running {precision.upper()} precision test...")
    print(f"{'='*50}")
    
    start_time = time.time()
    result = subprocess.run(['python', f'poly_{precision}.py'], 
                          capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Output: {result.stdout}")
    if result.stderr:
        print(f"Errors: {result.stderr}")
    print(f"Runtime: {end_time - start_time:.2f}s")

print(f"\n{'='*50}")
print("All precision tests completed!")
print("Generated files: poly_fp16.gif, poly_bf16.gif, poly_fp8.gif, poly_fp4.gif")
print(f"{'='*50}")