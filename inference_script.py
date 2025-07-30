from transformers import AutoTokenizer
from optimum.onnxruntime.modeling_seq2seq import ORTModelForSeq2SeqLM
import torch

# Load the tokenizer and model
model_id = "Mitchins/codet5-small-terminal-describer-quantized"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForSeq2SeqLM.from_pretrained(model_id)

def describe_command(command: str) -> str:
    """
    Generates a natural language description for a given shell command.
    """
    input_text = f"describe: {command}"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate description
    with torch.no_grad():
        outputs = model.generate(**inputs)

    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description

if __name__ == "__main__":
    print("T5 Terminal Command Describer (Quantized ONNX)")
    print("Enter 'exit' to quit.")

    # Example commands for sanity checking
    example_commands = [
        "ls -l",
        "git commit -m 'Initial commit'",
        "docker build -t my-image .",
        "kubectl get pods -n kube-system",
        "python my_script.py --input data.csv --output results.txt",
        "grep -r 'error' /var/log/",
        "sudo apt update && sudo apt upgrade",
        "ssh user@hostname 'ls -la /var/www'",
        "find . -name '*.txt' -exec rm {} \;",
        "tar -czvf archive.tar.gz /path/to/directory"
    ]

    print("\n--- Sanity Checks ---")
    for cmd in example_commands:
        print(f"Command: {cmd}")
        description = describe_command(cmd)
        print(f"Description: {description}\n")

    print("\n--- Interactive Mode ---")
    while True:
        command_input = input("Enter a shell command: ")
        if command_input.lower() == 'exit':
            break
        if command_input:
            description = describe_command(command_input)
            print(f"Description: {description}\n")
        else:
            print("Please enter a command.\n")
