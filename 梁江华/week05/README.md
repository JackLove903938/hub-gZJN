# week05 GPT-2 mini training

This folder now contains two scripts:

- `train.py` — trains `gpt2_min.py` on `corpus.txt` and saves a `.pt` checkpoint.
- `gen.py` — loads the `.pt` checkpoint and generates text from a user prompt.

## Train

```powershell
python train.py --max_steps 2000 --output_path gpt2_min.pt
```

For a quick smoke test:

```powershell
python train.py --max_steps 1 --eval_interval 1 --eval_iters 1 --output_path gpt2_min.pt
```

## Generate

```powershell
python gen.py --model_path gpt2_min.pt --prompt "Once upon a time" --max_new_tokens 100
```

## Notes

- The tokenizer is character-level, so the checkpoint stores the exact `stoi`/`itos` mapping used during training.
- Use the same `block_size` at generation time as the one used during training, because it is stored in the checkpoint configuration.

