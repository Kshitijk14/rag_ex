## [Qwen API Calls through Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3#transformers.Qwen3ForCausalLM)

```
from transformers import AutoTokenizer, Qwen3ForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = Qwen3ForQuestionAnswering.from_pretrained("Qwen/Qwen3-8B")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

# target is "nice puppet"
target_start_index = torch.tensor([14])
target_end_index = torch.tensor([15])

outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = outputs.loss
round(loss.item(), 2)
```