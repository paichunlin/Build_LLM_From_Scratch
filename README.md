### Build Larage Language Model from Scratch
Here is the model architecture diagram that I implemented:
<img width="801" height="542" alt="Screenshot 2025-12-01 at 9 24 08 PM" src="https://github.com/user-attachments/assets/ba762af9-654d-435f-b5c8-8e4b93657397" />


### Prepare data
Please follow the instructions to prepare the training and validation data:
1. Install uv
```
pip install uv run
```
2. download training and validation data
```
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

```
3. Install all packages
```
uv sync
```
4. Tokenize the data and save them in bin files
```
uv run python prepare.py \
    --input-path <path_to_file.txt> \
    --output-path <path_to_write_output.bin>
```

### Train the Model
Please follow the instructions to train the large language model:
1. Train the model
```
python train.py \
    --train-input-path <path_to_train.bin> \
    --test-input-path <path_to_test.bin>
```

