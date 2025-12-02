"""
tokenize data and save it in a bin file.

Running:

```
python prepare.py \
    --input-path <path_to_predictions.bin> \
    --output-path <path_to_write_output.bin>
```
"""
import argparse
import logging
import sys
from statistics import mean
from tqdm import tqdm
import os
from typing import BinaryIO
import numpy as np
import tiktoken
import json

logger = logging.getLogger(__name__)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_chunks(input_path: str, num_processes: int):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        chunks = []
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), desc="Read chunks"):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        return chunks

def write_token_ids(tokenizer, chunks: list[str]):
    total_len = 0
    with open("token_ids.txt", "a") as file:
        for text in tqdm(chunks, desc="Tokenizing"):
            encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
            total_len += len(encoded)
            file.write(json.dumps(encoded))
    
    return total_len

def write_to_bin_file(output_path: str, total_len: int):
    token_ids = np.memmap(output_path, dtype=np.int64, mode='w+', shape=(total_len,))
    with open("token_ids.txt", "r") as file:
        idx = 0
        for line in tqdm(file, desc="writing to bin"):
            encoded = json.loads(line)
            token_ids[idx : idx + len(line)] = encoded
            idx += len(line)
    token_ids.flush()
    
def main(
        input_path: str, 
        num_processes: int, 
        output_path: str
        ):
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    chunks = get_chunks(input_path, num_processes)
    total_len = write_token_ids(tokenizer, chunks)
    print("total_len=", total_len)
    write_to_bin_file(output_path, total_len)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to file with model predictions (JSONL format with key 'output')",
    )
    parser.add_argument(
        "--model-name-or-path", help="HF name of the model to use", required=True
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to write output predictions",
        required=True,
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.num_processes,
        args.model_name_or_path,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
