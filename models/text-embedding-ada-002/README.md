---
library_name: transformers
tags:
 - transformers.js
 - tokenizers
---

# text-embedding-ada-002 Tokenizer

A 🤗-compatible version of the **text-embedding-ada-002 tokenizer** (adapted from [openai/tiktoken](https://github.com/openai/tiktoken)). This means it can be used with Hugging Face libraries including [Transformers](https://github.com/huggingface/transformers), [Tokenizers](https://github.com/huggingface/tokenizers), and [Transformers.js](https://github.com/xenova/transformers.js).

## Example usage:

### Transformers/Tokenizers
```py
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/text-embedding-ada-002')
assert tokenizer.encode('hello world') == [15339, 1917]
```

### Transformers.js
```js
import { AutoTokenizer } from '@xenova/transformers';

const tokenizer = await AutoTokenizer.from_pretrained('Xenova/text-embedding-ada-002');
const tokens = tokenizer.encode('hello world'); // [15339, 1917]
```
