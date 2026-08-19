import torch
from compressed_tensors.quantization import QuantizationScheme
from compressed_tensors.quantization.quant_scheme import W4A16
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# NOTE: Qwen3.8 requires transformers >= 5.8.0.

MODEL_ID = "Qwen/Qwen3.8-27B"

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Quantize only the three dense MLP projections in the 64 text-transformer
# layers. Anchoring the full module path keeps the vision tower, linear/full
# attention, embeddings, lm_head, and the auxiliary MTP layer in BF16.
MLP_TARGETS = [
    r"re:^model\.language_model\.layers\.\d+\.mlp\."
    r"(gate_proj|up_proj|down_proj)$",
]

IGNORE = [
    r"re:.*lm_head$",
    r"re:.*visual\..*",
    r"re:.*embed_tokens$",
    r"re:.*\.self_attn\..*",
    r"re:.*\.linear_attn\..*",
    r"re:^mtp\..*",
]

# W4A16 is symmetric INT4 weight-only quantization with group size 128.
# Activations remain in the model's original 16-bit dtype (BF16 here).
recipe = GPTQModifier(
    config_groups={
        "text_mlp_w4a16": QuantizationScheme(
            targets=MLP_TARGETS,
            **W4A16,
        ),
    },
    ignore=IGNORE,
    block_size=128,
    dampening_frac=0.01,
    actorder=None,
    # Move Hessians to CPU between updates to reduce peak GPU memory usage.
    offload_hessians=True,
)

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.select_columns(["messages"])
ds = ds.shuffle(seed=42)


def preprocess_function(example):
    messages = [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in example["messages"]
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        processor_kwargs={
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": MAX_SEQUENCE_LENGTH,
            "add_special_tokens": False,
        },
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

# Qwen3_5ForConditionalGeneration does not load the MTP tensors. Save the
# compressed main model first, then copy the original unquantized MTP tensors
# into the output checkpoint so speculative decoding remains available.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MLP-W4A16-G128-GPTQ"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
