import argparse
import transformers

def main(args):
    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    # Resize token embeddings if necessary
    if tokenizer is not None and model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save model and tokenizer.")
    parser.add_argument("--model_name_or_path", type=str, help="Path or name of the pre-trained model.")
    parser.add_argument("--tokenizer_name_or_path", type=str, help="Path or name of the pre-trained tokenizer.")
    parser.add_argument("--output_dir", type=str, help="Directory where the model and tokenizer will be saved.")

    args = parser.parse_args()
    main(args)
