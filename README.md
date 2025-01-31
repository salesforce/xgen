# XGen

Official research release for the family of **XGen** models (`7B`) by Salesforce AI Research:

*Title*: [Long Sequence Modeling with XGen: A 7B LLM Trained on 8K Input Sequence Length](https://arxiv.org/abs/2309.03450)

*Authors*: [Erik Nijkamp](https://eriknijkamp.com)\*, Tian Xie\*, [Hiroaki Hayashi](https://hiroakih.me/)\*, [Bo Pang](https://scholar.google.com/citations?user=s9fNEVEAAAAJ&hl=en)\*, Congying Xia\*, Chen Xing, Jesse Vig, Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech Kryscinski, Lidiya Murakhovs'ka, Prafulla Kumar Choubey, Alex Fabbri, Ye Liu, Rui Meng, Lifu Tu, Meghana Bhat, [Chien-Sheng Wu](https://jasonwu0731.github.io/), Silvio Savarese, [Yingbo Zhou](https://scholar.google.com/citations?user=H_6RQ7oAAAAJ&hl=en), [Shafiq Rayhan Joty](https://raihanjoty.github.io/), [Caiming Xiong](http://cmxiong.com/).

(* indicates equal contribution)

Correspondence to: [Shafiq Rayhan Joty](mailto:sjoty@salesforce.com), [Caiming Xiong](mailto:cxiong@salesforce.com)

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
  author={Erik Nijkamp, Tian Xie, Hiroaki Hayashi, Bo Pang, Congying Xia, Chen Xing, Jesse Vig, Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech Kryscinski, Lidiya Murakhovs'ka, Prafulla Kumar Choubey, Alex Fabbri, Ye Liu, Rui Meng, Lifu Tu, Meghana Bhat, Chien-Sheng Wu, Silvio Savarese, Yingbo Zhou, Shafiq Rayhan Joty, Caiming Xiong},
  howpublished={ArXiv},
  year={2023},
  url={https://arxiv.org/abs/2309.03450}
}
```

## Ethics disclaimer for Salesforce AI models, data, code

This release is for research purposes only in support of an academic
paper. Our models, datasets, and code are not specifically designed or
evaluated for all downstream purposes. We strongly recommend users
evaluate and address potential concerns related to accuracy, safety, and
fairness before deploying this model. We encourage users to consider the
common limitations of AI, comply with applicable laws, and leverage best
practices when selecting use cases, particularly for high-risk scenarios
where errors or misuse could significantly impact people’s lives, rights,
or safety. For further guidance on use cases, refer to our standard
[AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ExternalFacing_Services_Policy.pdf)
and [AI AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ai-acceptable-use-policy.pdf).
