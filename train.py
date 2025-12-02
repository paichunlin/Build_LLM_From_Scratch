"""
train the model - pretraining stage

Running:

```
python train.py \
    --train-input-path <path_to_train.bin> \
    --test-input-path <path_to_test.bin>
```
depenedencies
train.py
 - data.py
 - model.py

model.py
 - nn_utils

"""

import argparse
import logging
import sys
import torch

from data import get_batch, estimate_loss
from model import BasicsTransformerLM
from nn_utils import cross_entropy, get_lr
import torch.optim as optim
import wandb

logger = logging.getLogger(__name__)

def main(train_input_path, 
         test_input_path,
         project_name,
         experiment_name,
         args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BasicsTransformerLM(args.vocab_size,
                                args.context_length,
                                args.d_model,
                                args.num_layers,
                                args.num_heads,
                                args.d_ff,
                                args.rope_theta)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters())
    config = {
        'batch_size': args.batch_size        
    }
    wandb.init(project=project_name, name=experiment_name, config=config)
    for num_iter in range(args.max_iters):
        input_ids, labels = get_batch(train_input_path, args.batch_size, args.context_length, device)
        logger.info("processing batch %i - size=%i", num_iter, len(input_ids))  
        input_ids = input_ids.to(device)
        logits = model(input_ids)

        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr(num_iter, 
                                       args.max_learning_rate,
                                       args.min_learning_rate,
                                       args.warmup_iters,
                                       args.cosine_cycle_iters)


        train_loss = cross_entropy(logits, labels)

        wandb.log({
            "iter": num_iter,
            "train/loss": train_loss.item(),
            })

        train_loss.backward()

        if (num_iter + 1) % args.gradient_accumulation_steps == 0:
            logger.info("optimizer step")  
            optimizer.step()
            optimizer.zero_grad()

        if num_iter % args.eval_interval == 0:
            validation_loss = estimate_loss(model, 
                                args.eval_iters,
                                test_input_path,
                                args.batch_size, 
                                args.context_length,
                                device)
            wandb.log({
                    "iter": num_iter,
                    "val/loss": validation_loss.item(),
            })        


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-input-path",
        type=str,
        required=True,
        help="Path to training tokens input file",
    )
    parser.add_argument(
        "--test-input-path",
        type=str,
        required=True,
        help="Path to test tokens input file",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Project name",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment name",
    )
    #Todo: gpt4 vocab size is 100000; gpt-4o vocab size is 200000? verify 200000 is a good number to use.
    parser.add_argument("--vocab-size", help="Vocabulary size", type=int, default=200000)
    parser.add_argument("--context-length", help="Context length of the model", type=int, default=4096)
    parser.add_argument("--d-model", help="The dimensionality of the model embeddings and sublayer outputs", type=int, default=768)
    parser.add_argument("--num-layers", help="Numer of transformer block layers", type=int, default=12)
    parser.add_argument("--num-heads", help="Numer of heads in transformer block", type=int, default=12)
    parser.add_argument("--d-ff", help="Dimensionality of the feed-forward inner layer", type=int, default=3072)
    parser.add_argument("--rope-theta", help="Context length of the model", type=float, default=10000.0)

    parser.add_argument("--batch-size", help="Batch size", type=int, default=128)
    parser.add_argument("--max-iters", help="Max training iterations", type=int, default=1000)
    parser.add_argument("--max-learning-rate", help="Max learning rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", help="Min learning rate", type=float, default=1e-5)
    parser.add_argument("--warmup-iters", help="Warmup iterations", type=int, default=2000)
    parser.add_argument("--cosine-cycle-iters", help="Min learning rate", type=int, default=600000)
    parser.add_argument("--eval-interval", help="Eval interval", type=int, default=200)
    parser.add_argument("--eval-iters", help="number of eval iterations", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", help="Gradient accumulation steps", type=int, default=1)

    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))

    main(
        args.train_input_path,
        args.test_input_path,
        args.project_name,
        args.experiment_name,
        args
    )
    logger.info("finished running %s", sys.argv[0])