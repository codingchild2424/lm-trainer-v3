# LM-Trainer-v3

This repo is for training LM easily.

## Env
- Nvida Docker Image is needed. Use Dockerfile.
- Use requirements.txt
```bash
pip install -U -r requirements.txt
```

or

- You can use huggingface docker image from Dockerhub
https://hub.docker.com/r/huggingface/transformers-all-latest-torch-nightly-gpu
- Install deepspeed
```bash
pip install -U deepspeed
```


## Experiements
- A100(VRAM 80GB) x 8


## Reference
- KoAlpaca (https://github.com/Beomi/KoAlpaca)
- trl (https://github.com/huggingface/trl)
- transformers (https://github.com/huggingface/transformers)