import math
import random
import numpy as np

def parse_cmdline_args(parser):
    parser.add_argument(
        '--l1_file',
        type=str,
        help='the path to the l1 score file.'
    )

    parser.add_argument(
        '--l2_file',
        type=str,
        help='the path to the l2 score file.'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        help='the path to the output file.'
    )

    args = parser.parse_args()
    return args

def load_data(file_path):
    data = {}
    all_scores = []
    with open(file_path) as f:
        for idx, line in enumerate(f):
            line_data = line.strip().split(',')
            query_id = line_data[0]
            doc_id = int(line_data[1])
            label = int(line_data[2])
            score = float(line_data[3])
            all_scores.append(score)
            doc_data = (doc_id, label, score)

            if query_id not in data.keys():
                data[query_id] = list()
            data[query_id].append(doc_data)

    # sort the list so that they arrange in the descending order of their scores
    for query_id in data.keys():
        data[query_id] = sorted(data[query_id], key=lambda x: x[2], reverse=True)

    return data, all_scores

def exclude_ids(data):
    exclude_ids = []
    for query_id in data.keys():
        label_set = set([d[1] for d in data[query_id]])
        if len(label_set) == 1 and label_set == {0}:
            # if 1 not in set([d[1] for d in data[query_id]]):   ## alternatively, if the label is binary, we can check if 1 is in the set.
            exclude_ids.append(query_id)
    for query_id in exclude_ids:
        del data[query_id]
    print(len(exclude_ids))
    return data

def split_query_ids(data, split_ratio=0.5):
    query_ids = list(data.keys())
    num_queries = len(query_ids)
    num_val = int (num_queries * split_ratio)
    val_idx = set(random.sample(range(num_queries), num_val))
    val_ids = [query_ids[idx] for idx in val_idx]
    test_ids = [query_ids[idx] for idx in range(num_queries) if idx not in val_idx]
    return val_ids, test_ids

def zip_data(l1_data, l2_data, ids):
    zipped_data = {}
    for id in ids:
        zipped_data[id] = (l1_data[id], l2_data[id])
    return zipped_data

def calc_l1_risk_for_query(docs_for_query, threshold, relevance_level=1):
    ground_truth_docs = get_ground_truth_above_l(docs_for_query, relevance_level)
    fetched_docs = set([doc[0] for doc in docs_for_query if doc[2] >= threshold])

    num_fetched = len(ground_truth_docs.intersection(fetched_docs))
    loss = 1 - num_fetched / (1.0 if len(ground_truth_docs) == 0 else len(ground_truth_docs))
    return (loss, fetched_docs)

def get_ground_truth_above_l(docs_for_query, relevance_level=1):
    relevant_docs = [doc[0] for doc in docs_for_query if doc[1] >= relevance_level]
    return set(relevant_docs)

def calc_retrieval_lambda(val_data, alpha, relevance_level = 1):
    lambda_val = 0.5
    delta = abs(pre_lambda_val  - lambda_val)
    precision = 0.00001
    M = len(val_data.keys())
    threshold = (M + 1) * alpha - 1
    while delta >= precision:
        total_loss = 0
        for query_id, (docs_for_query, _)  in val_data.items():
            total_loss += calc_l1_risk_for_query(docs_for_query, lambda_val, relevance_level)[0]
        if total_loss > threshold:
            lambda_val -= delta / 2
        elif total_loss < threshold:
            lambda_val += delta / 2
        else:
            break
        delta /= 2
    return lambda_val

def calc_l2_risk_for_query(l1_retrieved_docs, l2_ground_truth):
    denominator = sum([1.0/math.log2(i+2) for i in range(len(l2_ground_truth))])
    common_docs = set(l1_retrieved_docs).intersection(set(l2_ground_truth))
    if len(common_docs) == 0:
        return 1
    else:
        nominator = sum([1.0/math.log2(i+2) for i in range(len(common_docs))])
    return 1 - nominator / denominator

def get_max_retrieval_lambda(val_data, beta, level = 1):
    lambda_val = 0.5
    delta = abs(pre_lambda_val  - lambda_val)
    precision = 0.0005
    M = len(val_data.keys())
    threshold = (M + 1) * beta - 1
    while delta >= precision:
        total_l2_loss = 0
        for query_id, (docs_for_query, _) in val_data.items():
            l1_fetched_docs = set([doc[0] for doc in docs_for_query if doc[2] >= lambda_val])
            l2_ground_truth_docs = set([doc[0] for doc in docs_for_query if doc[1] >= level])
            min_l2_risk_for_query = calc_l2_risk_for_query(l1_fetched_docs, l2_ground_truth_docs)
            total_l2_loss += min_l2_risk_for_query

        if total_l2_loss > threshold:
            lambda_val -= delta / 2
        elif total_l2_loss < threshold:
            lambda_val += delta / 2
        else:
            break
        delta /= 2

    # calc l1 risk
    total_l1_loss = 0
    for query_id, (docs_for_query, _) in val_data.items():
        l1_risk_for_query, _ = calc_l1_risk_for_query(docs_for_query, lambda_val, level)
        total_l1_loss += l1_risk_for_query
    alpha = (total_l1_loss + 1)/(M + 1)

    return lambda_val, alpha

if __name__ == "__main__":
    # Argparse helper
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parse_cmdline_args(parser)

    l1_file_name = args.l1_file
    l2_file_name = args.l2_file
    output_file_name = args.output_file
    l1_data, all_scores = load_data(l1_file_name)
    l2_data, _ = load_data(l2_file_name)

    exclude_ids(l1_data)
    exclude_ids(l2_data)

    for query_id in l1_data.keys():
        l1_data[query_id] = [(d[0], d[1], math.tanh(d[2]/average_score)) for d in l1_data[query_id]]

    for query_id in l2_data.keys():
        l2_data[query_id] = [(d[0], d[1], (d[2]/100)) for d in l2_data[query_id]]

    val_ids, test_ids = split_query_ids(l1_data, 0.5)
    val_zipped_data = zip_data(l1_data, l2_data, val_ids)
    test_zipped_data = zip_data(l1_data, l2_data, test_ids)

    beta = 0.3
    iteration_times = 100
    max_l1_lambda, max_alpha = get_max_retrieval_lambda(val_zipped_data, 0.3, 3)
    level = 3
    precision = 0.0005
    alpha_vals = np.linspace(0, max_alpha, num=21)

    M = len(val_zipped_data.keys())
    threshold = (M + 1) * beta - 1
    summary_by_alpha = {}
    for iteration in range(iteration_times):
        val_ids, test_ids = split_query_ids(l1_data, 0.5)
        val_zipped_data = zip_data(l1_data, l2_data, val_ids)
        test_zipped_data = zip_data(l1_data, l2_data, test_ids)

        l1_lambda_list = []
        for alpha in alpha_vals:
            l1_lambda_list.append(calc_retrieval_lambda(val_zipped_data, alpha, level))
        print(l1_lambda_list)

        best_prediction_size = 10000000
        for idx, l1_lambda_val in enumerate(l1_lambda_list):
            alpha_val = alpha_vals[idx]
            pre_l2_lambda_val = 0
            l2_lambda_val = 0.5
            delta = abs(pre_l2_lambda_val  - l2_lambda_val)
            while delta >= precision:
                total_l2_loss = 0
                for query_id, (l1_docs_for_query, l2_docs_for_query) in val_zipped_data.items():
                    l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= l1_lambda_val])
                    l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= l2_lambda_val and doc[0] in l1_fetched_docs])
                    l2_ground_truth_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= level])
                    l2_risk_for_query = calc_l2_risk_for_query(l2_retained_docs, l2_ground_truth_docs)
                    total_l2_loss += l2_risk_for_query

                if total_l2_loss > threshold:
                    l2_lambda_val -= delta / 2
                elif total_l2_loss < threshold:
                    l2_lambda_val += delta / 2
                else:
                    break
                pre_l2_lambda_val = l2_lambda_val
                delta /= 2

            ## verify control on test data
            total_l1_size, total_l2_size = 0, 0
            total_l1_loss, total_l2_loss = 0, 0
            M_test = len(val_zipped_data.keys())
            for query_id, (l1_docs_for_query, l2_docs_for_query) in test_zipped_data.items():
                l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= l1_lambda_val])
                l1_ground_truth = set([doc[0] for doc in l1_docs_for_query if doc[1] >= level])
                total_l1_size += len(l1_fetched_docs)
                total_l1_loss += 1 - len(l1_ground_truth.intersection(l1_fetched_docs)) / (1.0 if len(l1_ground_truth) == 0 else len(l1_ground_truth))
                l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= l2_lambda_val and doc[0] in l1_fetched_docs])
                total_l2_size += len(l2_retained_docs)
                l2_ground_truth_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= level])
                l2_risk_for_query = calc_l2_risk_for_query(l2_retained_docs, l2_ground_truth_docs)
                total_l2_loss += l2_risk_for_query

            avg_l1_loss = (total_l1_loss + 1)/(M_test + 1)
            avg_l2_loss = (total_l2_loss + 1)/(M_test + 1)
            avg_l1_size = total_l1_size / M_test
            avg_l2_size = total_l2_size / M_test
            prediction_size = avg_l1_size + avg_l2_size
            if prediction_size < best_prediction_size:
                best_prediction_size = prediction_size
            print('{}:{}:{}:{}:{}:{}:{}'.format(prediction_size, avg_l1_size, avg_l2_size, l1_lambda_val, l2_lambda_val, avg_l1_loss, avg_l2_loss))
            if alpha_val not in summary_by_alpha.keys():
                summary_by_alpha[alpha_val] = list()
            summary_by_alpha[alpha_val].append((prediction_size, avg_l1_size, avg_l2_size, avg_l1_loss, avg_l2_loss))

    for alpha, results in summary_by_alpha.items():
        iteration_times = len(results)
        total_pred_size, total_l1_size, total_l2_size, total_l1_loss, total_l2_loss = 0, 0, 0, 0, 0
        for tuple in results:
            total_pred_size += tuple[0]
            total_l1_size += tuple[1]
            total_l2_size += tuple[2]
            total_l1_loss += tuple[3]
            total_l2_loss += tuple[4]
        print('{}:{}:{}:{}:{}:{}'.format(alpha,
                                         total_pred_size / iteration_times,
                                         total_l1_size / iteration_times,
                                         total_l2_size / iteration_times,
                                         total_l1_loss / iteration_times,
                                         total_l2_loss / iteration_times))

    iteration_times = 100
    level = 1
    precision = 0.0005
    num_lambda_steps = 11

    summary_by_beta = {}
    for beta in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        for iteration in range(iteration_times):
            val_ids, test_ids = split_query_ids(l1_data, 0.5)
            val_zipped_data = zip_data(l1_data, l2_data, val_ids)
            test_zipped_data = zip_data(l1_data, l2_data, test_ids)

            max_l1_lambda, _ = get_max_retrieval_lambda(val_zipped_data, beta, level)
            lambda_grid = np.linspace(0, max_l1_lambda, num=num_lambda_steps)
            best_prediction_size = 10000000
            best_l1_size, best_l2_size = 0, 0
            best_alpha, best_beta = 0, 0
            for l1_lambda_val in lambda_grid:
                pre_l2_lambda_val = 0
                l2_lambda_val = 0.5
                delta = abs(pre_l2_lambda_val  - l2_lambda_val)
                M = len(val_zipped_data.keys())
                threshold = (M + 1) * beta - 1
                while delta >= precision:
                    total_loss = 0
                    for query_id, (l1_docs_for_query, l2_docs_for_query) in val_zipped_data.items():
                        l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= l1_lambda_val])
                        l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= l2_lambda_val and doc[0] in l1_fetched_docs])
                        l2_ground_truth_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= level])
                        l2_risk_for_query = calc_l2_risk_for_query(l2_retained_docs, l2_ground_truth_docs)
                        total_loss += l2_risk_for_query

                    if total_loss > threshold:
                        l2_lambda_val -= delta / 2
                    elif total_loss < threshold:
                        l2_lambda_val += delta / 2
                    else:
                        break
                    pre_l2_lambda_val = l2_lambda_val
                    delta /= 2

                ## verify control on test data
                total_l1_size, total_l2_size = 0, 0
                total_l1_loss, total_l2_loss = 0, 0
                M_test = len(val_zipped_data.keys())
                for query_id, (l1_docs_for_query, l2_docs_for_query) in test_zipped_data.items():
                    l1_fetched_docs = set([doc[0] for doc in l1_docs_for_query if doc[2] >= l1_lambda_val])
                    l1_ground_truth = set([doc[0] for doc in l1_docs_for_query if doc[1] >= level])
                    total_l1_size += len(l1_fetched_docs)
                    total_l1_loss += 1 - len(l1_ground_truth.intersection(l1_fetched_docs)) / (1.0 if len(l1_ground_truth) == 0 else len(l1_ground_truth))
                    l2_retained_docs = set([doc[0] for doc in l2_docs_for_query if doc[2] >= l2_lambda_val and doc[0] in l1_fetched_docs])
                    total_l2_size += len(l2_retained_docs)
                    l2_ground_truth_docs = set([doc[0] for doc in l2_docs_for_query if doc[1] >= level])
                    l2_risk_for_query = calc_l2_risk_for_query(l2_retained_docs, l2_ground_truth_docs)
                    total_l2_loss += l2_risk_for_query

                avg_l1_loss = (total_l1_loss + 1)/(M_test + 1)
                avg_l2_loss = (total_l2_loss + 1)/(M_test + 1)
                avg_l1_size = total_l1_size / M_test
                avg_l2_size = total_l2_size / M_test
                prediction_size = avg_l1_size + avg_l2_size
                if prediction_size < best_prediction_size:
                    best_prediction_size = prediction_size
                    best_l1_size = avg_l1_size
                    best_l2_size = avg_l2_size
                    best_alpha = avg_l1_loss
                    best_beta = avg_l2_loss

            print('{}:{}:{}:{}:{}:{}'.format(beta, best_prediction_size, best_l1_size, best_l2_size, best_alpha, best_beta))
            if beta not in summary_by_beta.keys():
                summary_by_beta[beta] = list()
            summary_by_beta[beta].append((best_prediction_size, best_l1_size, best_l2_size, best_alpha, best_beta))

    for beta, results in summary_by_beta.items():
        iteration_times = len(results)
        total_pred_size, total_l1_size, total_l2_size, total_l1_loss, total_l2_loss = 0, 0, 0, 0, 0
        for tuple in results:
            total_pred_size += tuple[0]
            total_l1_size += tuple[1]
            total_l2_size += tuple[2]
            total_l1_loss += tuple[3]
            total_l2_loss += tuple[4]
        print('{}:{}:{}:{}:{}:{}'.format(beta, total_pred_size / iteration_times, total_l1_size / iteration_times,
                                         total_l2_size / iteration_times, total_l1_loss / iteration_times, total_l2_loss / iteration_times))