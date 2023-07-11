import MeCab

from argparse import ArgumentParser
from typing import List, Tuple


def load_mecab_data(path: str) -> List[Tuple[str, int]]:
    """
    単語とその出現回数を返す。
    """
    mecab = MeCab.Tagger("-Owakati")
    word_counts = dict() 
    with open(path) as fp:
        for line in fp:
            for word in mecab.parse(line.strip()).strip().split(' '):
                word_counts[word] = word_counts.get(word, 0) + 1
    
    return list(word_counts.items())


def construct_dictionary(
        data_path: str,
        vocabruary_size: int,
        merge_rule_output_path: str,
):
    word_counts = load_mecab_data(data_path)
    print("distinct num words: %s" % (len(word_counts),))
    print("total num words: %s" % (sum([count for _, count in word_counts])))
    print("word counts: %s" % (word_counts[: 5]))

    split_words = [(list(word), count) for word, count in word_counts]
    letters = set()
    for split_word, _ in split_words:
        for letter in split_word:
            letters.add(letter)
    print("distinct num letters: %d" % (len(letters),))

    pair_counts = dict() # トークンペアの出現回数を保持。
    pair_indexes = dict() # 高速化のためにトークンペアの出現する単語のインデックスを保持。
    for split_word_index, (split_word, count) in enumerate(split_words):
        for i in range(len(split_word) - 1):
            pair = tuple(split_word[i: i + 2])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_indexes.setdefault(pair, set()).add(split_word_index)

    merge_rules = []
    while len(letters) + len(merge_rules) < vocabruary_size:
        write_logging = (len(merge_rules) % 100 == 0)
        if write_logging:
            print("rule size: %d" % (len(merge_rules),))
        
        # 最も出現回数の多いトークンペアを取ってくる。
        maxi = max(pair_counts.values())
        for key, count in pair_counts.items():
            if count == maxi:
                most_frequent_pair = key
                break

        # ロード時にスペースで区切られてるかで単語かマージ規則かを判別するので間にスペースを入れる。
        merge_rules.append(most_frequent_pair)
        new_word = ''.join(most_frequent_pair)

        if write_logging:
            print("new word: %s, %d" % (new_word, pair_counts[most_frequent_pair]))
        
        # 該当のsplit_wordsをアップデートする。
        for split_word_index in pair_indexes[most_frequent_pair]:
            prev_word, count = split_words[split_word_index]

            # 該当箇所をマージする。
            next_word = []
            j = 0
            while j + 1 < len(prev_word):
                if prev_word[j] == most_frequent_pair[0] and prev_word[j + 1] == most_frequent_pair[1]:
                    # 高速化のため、マージ時にトークンペアの出現回数を更新しておく。
                    if 0 < j:
                        pair_counts[(next_word[-1], prev_word[j])] -= count
                        new_tuple = (next_word[-1], new_word)
                        pair_counts[new_tuple] = pair_counts.get(new_tuple, 0) + count
                        pair_indexes.setdefault(new_tuple, set()).add(split_word_index)
                    pair_counts[tuple(prev_word[j: j + 2])] -= count
                    if j + 2 < len(prev_word):
                        pair_counts[tuple(prev_word[j + 1: j + 3])] -= count
                        new_tuple = (new_word, prev_word[j + 2])
                        pair_counts[new_tuple] = pair_counts.get(new_tuple, 0) + count
                        pair_indexes.setdefault(new_tuple, set()).add(split_word_index)

                    next_word.append(new_word)
                    j += 2
                else:
                    next_word.append(prev_word[j])
                    j += 1
            if j < len(prev_word):
                next_word.append(prev_word[j])

            split_words[split_word_index] = (next_word, count)

    print("Saving to %s" % (merge_rule_output_path,))
    with open(merge_rule_output_path, 'w') as fp:
        # 文字はスペース区切りなしで書き込む。
        for letter in letters:
            fp.write("%s\n" % (letter,))
        # マージ規則はスペース区切りで書き込む。
        for rule in merge_rules:
            fp.write("%s\n" % (' '.join(rule),))


def run():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data-path', type=str)
    arg_parser.add_argument('--vocabruary-size', type=int)
    arg_parser.add_argument('--merge-rule-output-path', type=str)
    args = arg_parser.parse_args()
    print(args)

    construct_dictionary(
        data_path=args.data_path,
        vocabruary_size=args.vocabruary_size,
        merge_rule_output_path=args.merge_rule_output_path,
    )


if __name__ == '__main__':
    run()
