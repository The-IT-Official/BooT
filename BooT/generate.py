import argparse
import torch
import tiktoken
from model import BooT

def main():
    parser = argparse.ArgumentParser(description="Generate text using BooT LLM")
    parser.add_argument("--prompt", type=str, default="The history of artificial intelligence", help="Input prompt")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--max_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k filtering")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")

    model = BooT(vocab_size=enc.n_vocab).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)

        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state_dict)

    tokens = torch.tensor(enc.encode(args.prompt)).unsqueeze(0).to(device)
    out = model.generate(tokens, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
    print(enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()