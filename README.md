# XGen

Official research release for the family of **XGen** models (`7B`) by Salesforce AI Research:

*Title*: [Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length](https://blog.salesforceairesearch.com/xgen/)

*Authors*: [Erik Nijkamp](https://eriknijkamp.com)*, Tian Xie*, [Hiroaki Hayashi](https://hiroakih.me/)*, [Bo Pang](https://scholar.google.com/citations?user=s9fNEVEAAAAJ&hl=en)*, Congying Xia*, Chen Xing, Rui Meng, Wojciech Kryscinski, Lifu Tu, Meghana Bhat, Semih Yavuz, Jesse Vig, Lidiya Murakhovs'ka, [Chien-Sheng Wu](https://jasonwu0731.github.io/), [Yingbo Zhou](https://scholar.google.com/citations?user=H_6RQ7oAAAAJ&hl=en), [Shafiq Rayhan Joty](https://raihanjoty.github.io/), [Caiming Xiong](http://cmxiong.com/), Silvio Savarese.

(* indicates equal contribution).

## Models

Model cards are published on the HuggingFace Hub:

* [XGen-7B-4K-Base](https://huggingface.co/Salesforce/xgen-7b-4k-base) with support for 4K sequence length.
* [XGen-7B-8K-Base](https://huggingface.co/Salesforce/xgen-7b-8k-base) with support for 8K sequence length.
* [XGen-7B-8k-Inst](https://huggingface.co/Salesforce/xgen-7b-8k-inst) with instruction-finetuning (for research purpose only).

The tokenization uses the OpenAI Tiktoken package, which can be installed via `pip`:

```sh
pip install tiktoken
```

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
  author={Erik Nijkamp, Tian Xie, Hiroaki Hayashi, Bo Pang, Congying Xia, Chen Xing, Rui Meng, Wojciech Kryscinski, Lifu Tu, Meghana Bhat, Semih Yavuz, Jesse Vig, Lidiya Murakhovs'ka, Chien-Sheng Wu, Yingbo Zhou, Shafiq Rayhan Joty, Caiming Xiong, Silvio Savarese},
  howpublished={Salesforce AI Research Blog},
  year={2023},
  url={https://blog.salesforceairesearch.com/xgen}
}
```
