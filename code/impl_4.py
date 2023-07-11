sentences = [
    'this is readable',
    'read and click this clickable button'
]

# 全ての文字を辞書(dictionary)に追加する。
# 文字レベルに分割された単語をsplit_wordsに入れる。文字レベルなのは辞書に文字が入っているから。
# 後に単語が辞書に登録される度にsplit_words内の文字も単語レベルに結合されていく。
split_words = []
dictionary = set()
for sentence in sentences:
    for word in sentence.split(' '):
        split_words.append(list(word))
        dictionary.update(list(word))

# dictionary: {'u', 'r', 'a', 'i', 't', 'c', 'd', 'k', 'n', 'h', 'o', 's', 'b', 'l', 'e'}
# split word: ['t', 'h', 'i', 's']
# split word: ['i', 's']
# split word: ['r', 'e', 'a', 'd', 'a', 'b', 'l', 'e']
# split word: ['r', 'e', 'a', 'd']
# split word: ['a', 'n', 'd']
# split word: ['c', 'l', 'i', 'c', 'k']
# split word: ['t', 'h', 'i', 's']
# split word: ['c', 'l', 'i', 'c', 'k', 'a', 'b', 'l', 'e']
# split word: ['b', 'u', 't', 't', 'o', 'n']
print("dictionary: %s" % (dictionary,))
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
max_dictionary_size = 28
while len(dictionary) < max_dictionary_size:
    pair_counts = dict()
    for split_word in split_words:
        for i in range(len(split_word) - 1):
            pair = tuple(split_word[i: i + 2])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    most_frequent_pair = max(pair_counts, key=pair_counts.get)
    new_word = ''.join(most_frequent_pair)
    print("new word: %s, %d" % (new_word, pair_counts[most_frequent_pair]))
    dictionary.add(new_word)
    
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

# dictionary: {'i', 'able', 'cli', 'n', 'o', 'abl', 'e', 'u', 'a', 're', 'd', 'clic', 'ab', 'is', 's', 'cl', 'click', 'this', 'k', 'th', 'b', 'read', 'rea', 'r', 't', 'c', 'h', 'l'}
# split word: ['this']
# split word: ['is']
# split word: ['read', 'able']
# split word: ['read']
# split word: ['a', 'n', 'd']
# split word: ['click']
# split word: ['this']
# split word: ['click', 'able']
# split word: ['b', 'u', 't', 't', 'o', 'n']
print("dictionary: %s" % (dictionary,))
for split_word in split_words:
    print("split word: %s" % (split_word,))
