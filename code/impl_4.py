sentences = [
    'this is readable',
    'read and click this clickable button'
]

# 文字レベルに分割された単語をsplit_wordsに入れる。
# 後に、マージ規則が追加される度にsplit_words内の文字もマージされいていく。
split_words = []
for sentence in sentences:
    for word in sentence.split(' '):
        split_words.append(list(word))

# split word: ['t', 'h', 'i', 's']
# split word: ['i', 's']
# split word: ['r', 'e', 'a', 'd', 'a', 'b', 'l', 'e']
# split word: ['r', 'e', 'a', 'd']
# split word: ['a', 'n', 'd']
# split word: ['c', 'l', 'i', 'c', 'k']
# split word: ['t', 'h', 'i', 's']
# split word: ['c', 'l', 'i', 'c', 'k', 'a', 'b', 'l', 'e']
# split word: ['b', 'u', 't', 't', 'o', 'n']
for split_word in split_words:
    print("split word: %s" % (split_word,))

# 最も出現頻度の多いペアを辞書に追加していく。
# 1 回目: 'is'を追加
# 2 回目: 'th'を追加
# 3 回目: 'this'を追加
# 4 回目: 're'を追加
# .
# .
# .
# 13回目: 'click'を追加
merge_rules = []
while len(merge_rules) < 13:
    pair_counts = dict()
    for split_word in split_words:
        for i in range(len(split_word) - 1):
            pair = tuple(split_word[i: i + 2])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    most_frequent_pair = max(pair_counts, key=pair_counts.get)
    new_word = ''.join(most_frequent_pair)
    print("new word: %s, %d" % (new_word, pair_counts[most_frequent_pair]))
    merge_rules.append(most_frequent_pair)
    
    # split_wordsをアップデートする。
    for i, prev_word in enumerate(split_words):
        next_word = []
        j = 0
        while j < len(prev_word):
            if tuple(prev_word[j: j + 2]) == most_frequent_pair:
                next_word.append(new_word)
                j += 2
            else:
                next_word.append(prev_word[j])
                j += 1
        split_words[i] = next_word

# merge rules: [('i', 's'), ('t', 'h'), ('th', 'is'), ('r', 'e'), ('re', 'a'), ('rea', 'd'), ('a', 'b'), ('ab', 'l'), ('abl', 'e'), ('c', 'l'), ('cl', 'i'), ('cli', 'c'), ('clic', 'k')]
# split word: ['this']
# split word: ['is']
# split word: ['read', 'able']
# split word: ['read']
# split word: ['a', 'n', 'd']
# split word: ['click']
# split word: ['this']
# split word: ['click', 'able']
# split word: ['b', 'u', 't', 't', 'o', 'n']
print("merge rules: %s" % (merge_rules,))
for split_word in split_words:
    print("split word: %s" % (split_word,))
