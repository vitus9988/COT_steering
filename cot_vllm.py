import argparse
import os
import torch
import math
import pandas as pd
from typing import List, Tuple, Dict
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import login


###############################################################################
# Utility functions for token subsequence matching
###############################################################################
def find_all_subsequences(chosen_ids, target_ids):
    """
    Finds all subsequences in chosen_ids that match target_ids in consecutive order.
    Returns a list of (start, end) index pairs.
    If there is no match, returns an empty list [].
    """
    matches = []
    len_chosen = len(chosen_ids)
    len_target = len(target_ids)

    if len_target == 0 or len_chosen < len_target:
        return matches

    # Slide over all possible matching windows
    for start in range(len_chosen - len_target + 1):
        match = True
        for j in range(len_target):
            if chosen_ids[start + j] != target_ids[j]:
                match = False
                break
        if match:
            matches.append((start, start + len_target - 1))
    return matches


def find_last_subsequence(chosen_ids, target_ids):
    """
    Within chosen_ids, find the 'last match' (start, end) pair that corresponds
    to a consecutive occurrence of target_ids. If there's no match, returns (-1, -1).
    """
    matches = find_all_subsequences(chosen_ids, target_ids)
    if not matches:
        return (-1, -1)
    else:
        # If we have matches, return the last one
        return matches[-1]


def find_subsequence_with_spaces(chosen_ids, target_ids, tokenizer):
    """
    Attempt to find the last occurrence of target_ids in chosen_ids:
      1) Try matching target_ids directly.
      2) If that fails, decode target_ids, prepend a space (" ") to the front, re-encode, then try again.
      3) If that fails, decode target_ids, append a space to the end, re-encode, then try again.
      4) If no match is found, return (-1, -1).
    """
    # (1) Direct match
    idx_pair = find_last_subsequence(chosen_ids, target_ids)
    if idx_pair != (-1, -1):
        return idx_pair

    # Decode to string
    decoded_answer = tokenizer.decode(target_ids, skip_special_tokens=True)

    # (2) Prepend space, then re-encode
    decoded_answer_with_front_space = " " + decoded_answer
    target_ids_front_space = tokenizer.encode(decoded_answer_with_front_space, add_special_tokens=False)
    idx_pair = find_last_subsequence(chosen_ids, target_ids_front_space)
    if idx_pair != (-1, -1):
        return idx_pair

    # (3) Append space, then re-encode
    decoded_answer_with_back_space = decoded_answer + " "
    target_ids_back_space = tokenizer.encode(decoded_answer_with_back_space, add_special_tokens=False)
    idx_pair = find_last_subsequence(chosen_ids, target_ids_back_space)
    if idx_pair != (-1, -1):
        return idx_pair

    # (4) No match
    return (-1, -1)


###############################################################################
# 0) Model and tokenizer loading
###############################################################################
def load_model_and_tokenizer(
    model_name: str,
    tok_name: str,
    tensor_parallel_size: int,
    max_model_len: int
):
    model = LLM(
        model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.98,
        dtype='auto',
        enforce_eager=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tok_name,
        padding_side="left"
    )

    return model, tokenizer


###############################################################################
# 1) Prompt generation for the first pass
###############################################################################
def generate_prompt(prompt_text: str, tokenizer, steering_token: str) -> str:
    """
    Applies a chat template, then appends the given steering_token at the end.
    """
    messages = [
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    text += steering_token
    return text


###############################################################################
# 2) Extract top-k tokens in the first decoding step
###############################################################################
def get_top_k_first_tokens(model, prompts: List[str], k: int = 5) -> List[List[Tuple[str, float]]]:
    """
    For each prompt, obtains the top-k tokens and their log probabilities 
    in the first decoding step.
    """
    sampling_params = SamplingParams(
        max_tokens=1,
        logprobs=k
    )
    outputs = model.generate(prompts, sampling_params)

    top_k_for_all_prompts = []
    for req_output in outputs:
        if not req_output.outputs:
            top_k_for_all_prompts.append([])
            continue

        comp = req_output.outputs[0]
        if len(comp.logprobs) == 0:
            top_k_for_all_prompts.append([])
            continue

        # Retrieve logprobs for the first decoding step
        first_token_logprob_info = comp.logprobs[0]
        # Sort by rank in ascending order
        sorted_items = sorted(
            first_token_logprob_info.items(),
            key=lambda x: x[1].rank
        )
        # Take the top-k
        top_k_list = []
        for token_id, lp_info in sorted_items[:k]:
            tok_str = lp_info.decoded_token
            lp_val = lp_info.logprob
            top_k_list.append((tok_str, lp_val))

        top_k_for_all_prompts.append(top_k_list)

    return top_k_for_all_prompts


###############################################################################
# 3) Generate expanded prompts by appending top-k tokens
###############################################################################
def expand_prompts_with_topk(prompts: List[str], top_k_tokens_batch: List[List[Tuple[str, float]]]):
    """
    Given top_k_tokens_batch[i] = [(tok_str1, lp1), (tok_str2, lp2), ...],
    create new prompts by appending each token string to the original prompt.
    Also produce mapping_info = (i, tok_str, lp) to keep track of which
    original prompt index each new prompt originated from.
    """
    all_expanded_prompts = []
    mapping_info = []

    for i, prompt in enumerate(prompts):
        top_k_list = top_k_tokens_batch[i]
        for (tok_str, lp) in top_k_list:
            new_prompt = prompt + tok_str
            all_expanded_prompts.append(new_prompt)
            mapping_info.append((i, tok_str, lp))

    return all_expanded_prompts, mapping_info


###############################################################################
# 4) First-pass decoding (K generated answers) + storing logprobs
###############################################################################
def decode_multi_answer(
    model,
    prompts: List[str],
    k: int = 5,
    max_new_tokens: int = 1024,
    chunk_size: int = 1000
) -> List[Tuple[int, str, object]]:
    """
    Overall process:
      1) get_top_k_first_tokens -> expand_prompts_with_topk -> model.generate
      2) Return results as (prompt_idx, final_text, comp) 
         where comp includes vLLM's logprobs.
    """
    all_results = []

    for start_idx in tqdm(range(0, len(prompts), chunk_size)):
        prompt_chunk = prompts[start_idx : start_idx + chunk_size]

        # (A) Top-k for the first token
        top_k_tokens_batch = get_top_k_first_tokens(model, prompt_chunk, k=k)

        # (B) Create expanded prompts
        expanded_prompts, mapping_info = expand_prompts_with_topk(prompt_chunk, top_k_tokens_batch)

        # (C) Decode for actual answers
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            repetition_penalty=1.15,
            stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", 
                  "<|reserved_special_token|>","</s>","<|end_of_text|>", 
                  "Best regards",'<|im_end|>'],
            best_of=1,
            logprobs=2
        )
        outputs = model.generate(expanded_prompts, sampling_params)

        # (D) Store results in (global_idx, final_text, comp)
        for i, req_output in enumerate(outputs):
            local_idx, first_tok_str, first_tok_lp = mapping_info[i]
            global_idx = start_idx + local_idx

            if not req_output.outputs:
                # Decoding failure
                all_results.append((global_idx, "", None))
                continue

            comp = req_output.outputs[0]
            generated_partial = comp.text
            final_text = first_tok_str + generated_partial

            all_results.append((global_idx, final_text, comp))

    return all_results


###############################################################################
# 5) Creating second-pass prompts & extracting final answers
###############################################################################
def generate_extract_prompt_with_question(
    question_text: str,
    answer_text: str,
    tokenizer,
    extract_template: str,
    extract_token: str
) -> str:
    """
    Creates a prompt for the second-pass model. 
    This includes the original question and the full answer from the first pass (which may contain chain of thought).
    Then it applies the chat template, and appends the `extract_token` at the end 
    so that the second model can produce the final short answer.
    """
    # (1) Insert question + answer into the extract template
    filled_prompt = extract_template.format(
        question_text=question_text.strip(),
        answer_text=answer_text.strip(),
    )

    # (2) Apply chat template
    messages = [
        {"role": "user", "content": filled_prompt}
    ]
    prompt_for_model = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # (3) Append an extraction token trigger
    prompt_for_model += extract_token
    return prompt_for_model


def extract_final_answer_batch(
    model,
    questions: List[str],
    final_texts: List[str],
    tokenizer,
    extract_template: str,
    extract_token: str,
    chunk_size: int = 1000
) -> List[str]:
    """
    Takes multiple (question, first-pass answer) pairs in a batch and passes them
    to the second-pass model to extract the final short answer after `extract_token`.
    """
    batch_prompts = []
    for question, answer_text in zip(questions, final_texts):
        # If the first-pass answer is empty, just use an empty prompt
        if not answer_text.strip():
            batch_prompts.append("")
            continue

        extract_prompt = generate_extract_prompt_with_question(
            question,
            answer_text,
            tokenizer,
            extract_template,
            extract_token
        )
        batch_prompts.append(extract_prompt)

    # We generate answers in chunks to handle large input sizes
    extracted_answers = []
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", 
              "<|reserved_special_token|>","</s>","<|end_of_text|>", 
              "Best regards",'<|im_end|>'],
        logprobs=0
    )

    for start_i in tqdm(range(0, len(batch_prompts), chunk_size), desc="2nd Model Inference"):
        chunk_prompts = batch_prompts[start_i : start_i + chunk_size]
        outputs = model.generate(chunk_prompts, sampling_params)

        for i, out in enumerate(outputs):
            # If the prompt was empty, the output is also empty
            if (not out.outputs) or (not chunk_prompts[i].strip()):
                extracted_answers.append("")
            else:
                ans = out.outputs[0].text.strip()
                extracted_answers.append(ans)

    return extracted_answers


###############################################################################
# 6) Token ID-based confidence for extracted answer
###############################################################################
def calculate_confidence_for_extracted_answer_tokenwise(
    comp,               # vLLM result object that includes logprobs
    extracted_answer: str,
    tokenizer
) -> float:
    """
    (A) Tokenize extracted_answer -> target_ids
    (B) From comp, gather the top-1 IDs (chosen_ids) and their logprobs at each step,
        as well as the second-best logprobs.
    (C) Find the subsequence of chosen_ids that matches target_ids. (Take the last match.)
    (D) Compute the average of (top1_prob - top2_prob) for that matched token range.
    (E) Return the computed confidence score.
    """
    if not extracted_answer.strip():
        return 0.0

    target_ids = tokenizer.encode(extracted_answer, add_special_tokens=False)
    if not target_ids:
        return 0.0

    chosen_ids = []
    chosen_lp = []
    second_lp = []
    for token_step_dict in comp.logprobs:
        # token_step_dict: { token_id: LogprobsInfo(...) }
        sorted_candidates = sorted(token_step_dict.items(), key=lambda x: x[1].rank)
        if len(sorted_candidates) == 0:
            continue

        # top-1
        top1_id, top1_info = sorted_candidates[0]
        chosen_ids.append(top1_id)
        chosen_lp.append(top1_info.logprob)

        # top-2 (or placeholder if none)
        if len(sorted_candidates) > 1:
            top2_info = sorted_candidates[1][1]
            second_lp.append(top2_info.logprob)
        else:
            second_lp.append(-9999.0)

    # Find the matching subsequence (last occurrence)
    start_idx, end_idx = find_subsequence_with_spaces(chosen_ids, target_ids, tokenizer)
    if start_idx == -1:
        return 0.0

    total_diff = 0.0
    valid_count = 0
    for i in range(start_idx, end_idx + 1):
        top1_prob = math.exp(chosen_lp[i])
        top2_prob = math.exp(second_lp[i])
        diff = (top1_prob - top2_prob)
        total_diff += diff
        valid_count += 1

    if valid_count == 0:
        return 0.0
    return total_diff / valid_count


###############################################################################
# 7) Aggregation: group by identical extracted answers -> choose the best
###############################################################################
def aggregate_all_answer_groups(
    results_all,
):
    """
    Takes a list of (p_idx, final_text, extracted_answer, conf) and groups them by p_idx.
    Then, for each p_idx, groups by identical extracted_answer to sum up their confidence.
    Finally, picks the group with the highest sum_conf as the best group.
    """
    grouped_by_prompt = defaultdict(list)
    for (p_idx, f_text, e_ans, conf) in results_all:
        grouped_by_prompt[p_idx].append((f_text, e_ans, conf))

    aggregated = {}
    for p_idx, row_list in grouped_by_prompt.items():
        # row_list = [(final_text, e_ans, conf), ...]

        # 1) Group by extracted_answer
        answer_map = defaultdict(list)
        for (f_text, e_ans, c) in row_list:
            answer_map[e_ans].append((f_text, c))

        # 2) Calculate sum_conf for each answer
        answer_groups = []
        for e_ans, pair_list in answer_map.items():
            # pair_list = [(final_text, conf), ...]
            sum_conf = sum(x[1] for x in pair_list)
            # Sort in descending order by conf
            pair_list_sorted = sorted(pair_list, key=lambda x: x[1], reverse=True)

            answer_groups.append({
                'extracted_answer': e_ans,
                'sum_conf': sum_conf,
                'details': pair_list_sorted
            })

        # 3) Sort the groups by sum_conf, descending
        answer_groups_sorted = sorted(answer_groups, key=lambda x: x['sum_conf'], reverse=True)

        # 4) The best group is the first one after sorting if any exist
        if len(answer_groups_sorted) > 0:
            best_group = answer_groups_sorted[0]
            best_extracted_answer = best_group['extracted_answer']
            best_conf_sum = best_group['sum_conf']
        else:
            best_group = None
            best_extracted_answer = ""
            best_conf_sum = 0.0

        aggregated[p_idx] = {
            'all_groups_sorted': answer_groups_sorted,
            'best_extracted_answer': best_extracted_answer,
            'best_conf_sum': best_conf_sum,
            'best_group': best_group
        }

    return aggregated


###############################################################################
# main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--SAVE_DIR", type=str, default="result.csv")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tok_name", type=str, required=True)
    parser.add_argument("--template_path", type=str, required=True,
                        help="Path to the second-pass extraction template file.")
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--num_k", type=int, default=20, 
                        help="Top-k for the first token in first-pass decoding.")
    parser.add_argument("--df_path", type=str, required=True,
                        help="Path to the original CSV file containing prompts.")
    parser.add_argument("--STEERING_TOKEN", type=str, default="",
                        help="Token appended at the end of the first-pass prompts.")
    parser.add_argument("--EXTRACT_TOKEN", type=str, default="",
                        help="Token for extracted Signal")
    
    args = parser.parse_args()

    # 1) Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.tok_name,
        args.tensor_parallel_size,
        args.max_model_len
    )

    # 2) Load the extraction template
    with open(args.template_path, 'r', encoding='utf-8') as file:
        extract_template = file.read()

    EXTRACT_TOKEN = args.EXTRACT_TOKEN

    # 3) Load CSV
    df = pd.read_csv(args.df_path)

    # 4) Generate first-pass prompts
    prompts = [
        generate_prompt(prompt_text=x, tokenizer=tokenizer, steering_token=args.STEERING_TOKEN)
        for x in df['prompts'].tolist()
    ]

    # 5) Perform first-pass decoding -> (prompt_idx, final_text, comp)
    decoding_results = decode_multi_answer(
        model=model,
        prompts=prompts,
        k=args.num_k,
        max_new_tokens=16384,  # Adjust as needed
        chunk_size=args.batch_size
    )

    # 6) For second-pass decoding, gather final_texts and corresponding questions
    final_texts_all = []
    questions_for_2nd_inference = []
    for (p_idx, final_text, comp) in decoding_results:
        final_texts_all.append(final_text)
        questions_for_2nd_inference.append(df['prompts'].iloc[p_idx])

    # 7) Use second-pass model calls to extract final answers
    extracted_answers_all = extract_final_answer_batch(
        model=model,
        questions=questions_for_2nd_inference,
        final_texts=final_texts_all,
        tokenizer=tokenizer,
        extract_template=extract_template,
        extract_token=EXTRACT_TOKEN,
        chunk_size=args.batch_size
    )

    # 8) Compute confidence for each final extracted answer
    final_results = []
    for i, (p_idx, final_text, comp) in enumerate(decoding_results):
        extracted_answer = extracted_answers_all[i].strip()
        if comp is None or not final_text.strip():
            final_results.append((p_idx, final_text, "", 0.0))
            continue

        conf_val = calculate_confidence_for_extracted_answer_tokenwise(
            comp,
            extracted_answer,
            tokenizer
        )
        final_results.append((p_idx, final_text, extracted_answer, conf_val))

    # 9) Aggregate: group by identical answers -> pick the one with the highest sum_conf
    aggregated = aggregate_all_answer_groups(final_results)

    # 10) Store the best extracted answer, confidence, and group info back into df
    best_extracted_answers = []
    best_conf_sums = []
    all_groups_info = []
    best_group_info = []

    for i in range(len(df)):
        info = aggregated.get(i)
        if not info:
            best_extracted_answers.append("")
            best_conf_sums.append(0.0)
            all_groups_info.append([])
            best_group_info.append(None)
        else:
            best_extracted_answers.append(info['best_extracted_answer'])
            best_conf_sums.append(info['best_conf_sum'])
            all_groups_info.append(info['all_groups_sorted'])
            best_group_info.append(info['best_group'])

    df["best_extracted_answer"] = best_extracted_answers
    df["best_conf_sum"] = best_conf_sums
    df["best_group"] = best_group_info
    df["all_answer_groups"] = all_groups_info

    df.to_csv(args.SAVE_DIR, index=False)
    print(f"Saved results to {args.SAVE_DIR}")


if __name__ == "__main__":
    main()
