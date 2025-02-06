import subprocess
import keyboard  # Install with `pip install keyboard`

# Paths to your ML scripts
script1 = r"D:\OCR_raspberry_pi_hardware\OCR.py"
script2 = r"D:\OCR_raspberry_pi_hardware\ob.py"

try:
    # Start both scripts as separate processes
    print("Starting OCR.py...")
    process1 = subprocess.Popen(["python", script1], shell=True)

    print("Starting ob.py...")
    process2 = subprocess.Popen(["python", script2], shell=True)

    print("Both programs are running. Press 'q' to terminate them.")

    # Wait for 'q' key press to terminate both processes
    while True:
        if keyboard.is_pressed('q'):  # Monitor for 'q' key press
            print("\n'q' pressed. Terminating both programs...")
            process1.terminate()  # Terminate OCR.py
            process2.terminate()  # Terminate ob.py
            break

    print("Both programs have been terminated.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Ensure processes are terminated
    if process1.poll() is None:
        process1.terminate()
    if process2.poll() is None:
        process2.terminate()
