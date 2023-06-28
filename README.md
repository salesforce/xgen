# XGen

Official research release for the family of **XGen** models (`7B`) by Salesforce AI Research:

*Title*: [Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length](https://blog.salesforceairesearch.com/xgen-7b/)

*Authors*: [todo](https://todo/).

## Usage

Model cards are published on the HuggingFace Hub:

* [xgen-7b-4k-base](https://huggingface.co/Salesforce/xgen-7b-4k-base)
* [xgen-7b-8k-base](https://huggingface.co/Salesforce/xgen-7b-8k-base)

The models can be used as auto-regressive samplers as follows:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Salesforce/xgen-7b-8k-base", torch_dtype=torch.bfloat16)
inputs = tokenizer("The world is", return_tensors="pt")
sample = model.generate(**inputs, max_length=128)
print(tokenizer.decode(sample[0]))
```

## Citation

```bibtex
@misc{XGen,
  title={Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length},
  author={Salesforce AI Research},
  howpublished={Salesforce AI Research Blog},
  year={2023},
  url={https://blog.salesforceairesearch.com/xgen-7b/}
}
```
