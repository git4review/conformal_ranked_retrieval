import argparse
import json

import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def load_data(file_name):
    f = open (file_name, "r")
    data = json.loads(f.read())
    f.close()
    return data

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def calc_scores(tokenizer, model, input_texts):
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = ((embeddings[:1] @ embeddings[1:].T) * 100).tolist()
    return scores

def parse_cmdline_args(parser):
    parser.add_argument(
        '--filetype',
        type=str,
        choices=['train', 'dev'],
        default='train',
        help='which type of file do you want to load.'
    )

    parser.add_argument(
        '--input_file',
        type=str,
        help='the path to the input file.'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        help='the path to the output file.'
    )

    parser.add_argument(
        '--idx',
        type=int,
        default=0,
        help='which file shard do you want to load'
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Argparse helper
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parse_cmdline_args(parser)

    # load data
    input_file_name = args.input_file
    output_file_name = args.output_file
    data = load_data(input_file_name)

    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    model = AutoModel.from_pretrained('intfloat/e5-base-v2')

    f = open(output_file_name, 'a')
    for query_id in data["query"].keys():
        query = data["query"][query_id]

        input_texts = ['query: ' + query]
        passages = data["passages"][query_id]
        for idx, snippet in enumerate(passages):
            is_selected = snippet["is_selected"]
            passage_text = snippet["passage_text"]
            input_texts.append('passage: ' + passage_text)

        scores = calc_scores(tokenizer, model, input_texts)
        for idx, s in enumerate(scores[0]):
            output = '{}, {}, {}, {}\n'.format(query_id, idx, passages[idx]["is_selected"], s)
            f.write(output)

    f.close()