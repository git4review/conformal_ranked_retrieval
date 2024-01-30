import math
import json


def parse_cmdline_args(parser):
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

    args = parser.parse_args()
    return args

'''
the implementation follows the description in
https://en.wikipedia.org/wiki/Okapi_BM25#:~:text=6%20External%20links-,The%20ranking%20function,slightly%20different%20components%20and%20parameters.
'''
if __name__ == "__main__":
    # Argparse helper
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parse_cmdline_args(parser)

    input_file_name = args.input_file
    output_file_name = args.output_file

    # Reading from file
    f = open (input_file_name, "r")
    data = json.loads(f.read())
    f.close()

    q_doc_count_dict = {}
    q_idf_dict = {}
    total_doc_count = 0
    total_doc_len = 0
    k1 = 0.2
    b = 0.75

    for query_id in data["query"].keys():
        query = data["query"][query_id]
        passages = data["passages"][query_id]

        for idx, snippet in enumerate(passages):
            total_doc_count += 1
            passage_text = snippet["passage_text"].lower()
            passage_words = passage_text.split()
            passage_distinct_words = set(passage_words)
            for word in passage_distinct_words:
                if word in q_doc_count_dict.keys():
                    q_doc_count_dict[word] = 1 + q_doc_count_dict[word]
                else:
                    q_doc_count_dict[word] = 1
            doc_len = len(passage_words)
            total_doc_len += doc_len

    avg_doc_len = (total_doc_len + 0.0) / total_doc_count

    for q,count in q_doc_count_dict.items():
        idf = math.log((total_doc_count - count + 0.5) / (count + 0.5) + 1)
        q_idf_dict[q] = idf

    f = open(output_file_name, 'a')
    for query_id in data["query"].keys():
        query = data["query"][query_id].lower()
        query_words = set(query.split())
        passages = data["passages"][query_id]

        for idx, snippet in enumerate(passages):
            is_selected = snippet["is_selected"]
            passage_text = snippet["passage_text"].lower()
            passage_words = passage_text.split()
            doc_len = len(passage_words)
            score = 0.0
            for q_word in query_words:
                if q_word in q_idf_dict.keys():
                    f_qd = passage_words.count(q_word)
                    score += q_idf_dict[q_word] * f_qd * (k1 + 1) / (f_qd + k1 * (1 - b + b * doc_len / avg_doc_len))
            output = '{}, {}, {}, {}\n'.format(query_id, idx, is_selected, score)
            f.write(output)

    f.close()