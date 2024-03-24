# Gemma-2B Fine-Tuned Python Model
![OIP](https://github.com/Mr-Vicky-01/tamil_summarization/assets/143078285/d2e4fb7a-53d5-4e83-886f-7a8b5fc50504)

## Overview
Gemma-2B Fine-Tuned Python Model is a deep learning model based on the Gemma-2B architecture, fine-tuned specifically for Python programming tasks. This model is designed to understand Python code and assist developers by providing suggestions, completing code snippets, or offering corrections to improve code quality and efficiency.

## Model Details
- **Model Name**: Gemma-2B Fine-Tuned Python Model
- **Model Type**: Deep Learning Model
- **Base Model**: Gemma-2B
- **Language**: Python
- **Task**: Python Code Understanding and Assistance

## Example Use Cases
- Code completion: Automatically completing code snippets based on partial inputs.
- Syntax correction: Identifying and suggesting corrections for syntax errors in Python code.
- Code quality improvement: Providing suggestions to enhance code readability, efficiency, and maintainability.
- Debugging assistance: Offering insights and suggestions to debug Python code by identifying potential errors or inefficiencies.

## How to Use
1. **Install Gemma Python Package**:
   ```bash
    pip install -q -U transformers==4.38.0
    pip install torch
   ```

## Inference
1. **How to use the model in our notebook**:
```python
# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Mr-Vicky-01/Gemma-2B-Finetuined-pythonCode")
model = AutoModelForCausalLM.from_pretrained("Mr-Vicky-01/Gemma-2B-Finetuined-pythonCode")

query = input('enter a query:')
prompt_template = f"""
<start_of_turn>user based on given instruction create a solution\n\nhere are the instruction {query}
<end_of_turn>\n<start_of_turn>model
"""
prompt = prompt_template
encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = encodeds.to(device)


# Increase max_new_tokens if needed
generated_ids = model.generate(inputs, max_new_tokens=1000, do_sample=False, pad_token_id=tokenizer.eos_token_id)
ans = ''
for i in tokenizer.decode(generated_ids[0], skip_special_tokens=True).split('<end_of_turn>')[:2]:
    ans += i

# Extract only the model's answer
model_answer = ans.split("model")[1].strip()
print(model_answer)
```

## Output ScreenShot

![image](https://github.com/Mr-Vicky-01/tamil_summarization/assets/143078285/dcf7edab-0e24-4b5f-89bb-9a98059b7097)

## Model Link

[Hugging Face](https://huggingface.co/Mr-Vicky-01/Gemma-2B-Finetuined-pythonCode)
