# XGen

Official research release for the family of **XGen** models (`7B`) by Salesforce AI Research:

*Title*: [Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length](https://blog.salesforceairesearch.com/xgen/)

*Authors*: [todo](https://todo/).

## Usage

Model cards are published on the HuggingFace Hub:

* [xGen-7B](https://huggingface.co/Salesforce/xgen-7B)

The models can be used as auto-regressive samplers as follows:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("checkpoints/xgen-7B")
model = AutoModelForCausalLM.from_pretrained("checkpoints/xgen-7B", torch_dtype=torch.bfloat16, revision="sharded")
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
