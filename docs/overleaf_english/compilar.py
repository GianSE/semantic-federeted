import subprocess
import os

def compile_paper(main_file="main.tex"):
    # Get the base name without extension (e.g., "main")
    base_name = os.path.splitext(main_file)[0]

    # Check if the .tex file exists in the directory
    if not os.path.exists(main_file):
        print(f"Error: File '{main_file}' was not found in this directory.")
        return False

    # List of commands to run in order
    commands = [
        ["pdflatex", "-interaction=nonstopmode", main_file],
        ["bibtex", base_name],
        ["pdflatex", "-interaction=nonstopmode", main_file],
        ["pdflatex", "-interaction=nonstopmode", main_file]
    ]

    print(f"Starting compilation of: {main_file}\n" + "-"*40)

    for i, cmd in enumerate(commands, start=1):
        cmd_str = " ".join(cmd)
        print(f"[{i}/{len(commands)}] Running: {cmd_str}")
        
        try:
            # Execute the command
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Check for error (return code != 0)
            if result.returncode != 0:
                print(f"\n[!] An error occurred at step {i}.")
                print("Last lines of error output:")
                # Show last lines of log to help identify error
                output_lines = result.stdout.split('\n')
                print('\n'.join(output_lines[-20:])) 
                return False
                
            print("OK.")
            
        except FileNotFoundError:
            print(f"\n[!] Critical error: Command '{cmd[0]}' was not found.")
            print("Make sure MiKTeX, TeX Live or MacTeX is installed and configured in your environment variables (PATH).")
            return False

    print("-" * 40)
    print(f"Success! The paper was compiled and '{base_name}.pdf' was updated.")
    return True

if __name__ == "__main__":
    compile_paper("main.tex")