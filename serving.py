from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from huggingface_hub import login


HF_KEY = ''

login(token=HF_KEY)

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
llm = LLM(
    model=model_id,
    enable_lora=True,
)



lora_request = LoRARequest(
    lora_name='lora',
    lora_int_id=1,
    lora_path="sft_output/lora_adapter_only",
)



# from vllm import ChatCompletionRequestMessage
prompts = 'Write The Article Body.'
system_message = "You are an expert in extracting tags from Arabic news articles. Your task is to read the provided article text and extract the most relevant keywords that represent the main topics of the article. Return a list of the extracted keywords."

conversation = [
    {
        "role": "system",
        "content": system_message
    },
    {
        "role": "user",
        "content": prompts
    },
]


sampling_params = SamplingParams(
    temperature=0,
    max_tokens=64
)

outputs = llm.chat(conversation,
              sampling_params=sampling_params,
              lora_request=lora_request)



for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")