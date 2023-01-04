S1 = '나는 책상 위에 사과를 먹었다'
S2 = '알고 보니 그 사과는 Jason 것이었다'
S3 = '그래서 Jason에게 사과를 했다'

print(S1.split())
print(S2.split())
print(S3.split())
print(list(S1))


def indexed_sentence(sentence):
    return [token2idx[token] for token in sentence]


def indexed_sentence_unk(sentence):
    return [token2idx.get(token, token2idx['<unk>']) for token in sentence]


token2idx = {}
index = 0
for sentence in [S1, S2, S3]:
    tokens = sentence.split()
    for token in tokens:
        if token2idx.get(index) == None:
            token2idx[token] = index
            index += 1

print(token2idx)

S1_i = indexed_sentence(S1.split())
print(S1_i)

S2_i = indexed_sentence(S2.split())
print(S2_i)

S3_i = indexed_sentence(S3.split())
print(S3_i)

# Corpus & OOV
S4 = '나는 책상 위에 배를 먹었다'
# indexed_sentence(S4.split()) -> KeyError: '배를' 이라는 텍스트는 처음 보는 텍스트 이므로 확인이 불가능

token2idx = {t: i + 1 for t, i in token2idx.items()}
token2idx['<unk>'] = 0                                      # unk means unknow
indexed_sentence_unk(S4.split())
