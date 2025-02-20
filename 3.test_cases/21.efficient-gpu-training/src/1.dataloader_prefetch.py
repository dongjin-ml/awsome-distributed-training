import torch
from trl import TrlParser
from model_utils.train_utils import get_logger
from model_utils.train_utils import print_summary
from model_utils.train_utils import print_gpu_utilization

from model_utils.arguments import CustomArguments
from transformers import TrainingArguments, Trainer


from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import Dataset

import bitsandbytes as bnb

logger = get_logger()


if __name__ == "__main__":

    parser = TrlParser((CustomArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()


    logger.info("===")
    logger.info("Scripts Arguments: %s", vars(script_args))
    logger.info("Training Arguments: %s", vars(training_args))

    # print ("1script_args", script_args)
    # print ("1==")
    # print ("1training_args", training_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.ones((1, 1)).to(device)



    seq_len, dataset_size = 512, 512
    dummy_data = {
        "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
        "labels": np.random.randint(0, 1, (dataset_size)),
    }
    ds = Dataset.from_dict(dummy_data)
    ds.set_format("pt")

    model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to(device)
    print_gpu_utilization()

    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    result = trainer.train()
    print_summary(result)






# seq_len, dataset_size = 512, 512
# dummy_data = {
#     "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
#     "labels": np.random.randint(0, 1, (dataset_size)),
# }
# ds = Dataset.from_dict(dummy_data)
# ds.set_format("pt")

# dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

# if training_args.gradient_checkpointing:
#     model.gradient_checkpointing_enable()

# model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
# print_gpu_utilization()

# if training_args.fp16:
#     mixed_precision = "fp16"
# else:
#     mixed_precision = None

# accelerator = Accelerator(
#     mixed_precision=mixed_precision
# )
# adam_bnb_optim = bnb.optim.Adam8bit(
#     optimizer_grouped_parameters,
#     betas=(training_args.adam_beta1, training_args.adam_beta2),
#     eps=training_args.adam_epsilon,
#     lr=training_args.learning_rate,
# )

# model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

# model.train()


# for step, batch in enumerate(dataloader, start=1):
#     loss = model(**batch).loss
#     loss = loss / training_args.gradient_accumulation_steps
#     accelerator.backward(loss)
#     if step % training_args.gradient_accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()