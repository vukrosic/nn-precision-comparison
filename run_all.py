import subprocess
import time

precisions = ['fp16', 'bf16', 'fp8', 'fp4', 'fp4_bnb', 'fp4_sim']

for precision in precisions:
    print(f"Running {precision.upper()}...")
    start = time.time()
    result = subprocess.run(['python', f'poly_{precision}.py'], capture_output=True, text=True)
    runtime = time.time() - start
    
    if result.returncode == 0:
        # Extract final loss from output
        lines = result.stdout.strip().split('\n')
        final_line = lines[-1] if lines else ""
        print(f"{precision.upper()}: {final_line} ({runtime:.1f}s)")
    else:
        print(f"{precision.upper()}: FAILED - {result.stderr.strip()}")

print("\nGenerated GIFs for all precision types")