import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

train_sentences_tagged = open("macmorpho-train.txt", "r")
validation_sentences_tagged = open("macmorpho-dev.txt", "r")
test_sentences_tagged = open("macmorpho-test.txt", "r")

train_sentences, train_tags = [], [] #corpus todo de treino
for sentence in train_sentences_tagged:
    sent, tags = [], []#cada sentença
    sentence = sentence.split()#tem cada palavra+TAG da sentença em uma lista
    for word in sentence:
        word = word.split("_")
        sent.append(word[0])
        tags.append(word[1])
    train_sentences.append(sent)
    train_tags.append(tags)

print(train_sentences[0]) #teste so pra ver se separou certinho
print(train_tags[0])
    
valid_sentences, valid_tags = [], []
for sentence in validation_sentences_tagged:
    sent, tags = [], []#cada sentença
    sentence = sentence.split()#tem cada palavra+TAG da sentença em uma lista
    for word in sentence:
        word = word.split("_")
        sent.append(word[0])
        tags.append(word[1])
    valid_sentences.append(sent)
    valid_tags.append(tags)

test_sentences, test_tags = [], [] #corpus todo de treino
for sentence in test_sentences_tagged:
    sent, tags = [], []#cada sentença
    sentence = sentence.split()#tem cada palavra+TAG da sentença em uma lista
    for word in sentence:
        word = word.split("_")
        sent.append(word[0])
        tags.append(word[1])
    test_sentences.append(sent)
    test_tags.append(tags)

#Frase original: Salto_N sete_ADJ

words, tags = set([]), set([]) #sets que armazenam todas as palavras e tags que aparecem no treino
 
for sentence in train_sentences:
    for word in sentence:
        words.add(word.lower())
    
for sentence in train_tags:
    for tag in sentence:
        tags.add(tag)
        
word2index = {word: i + 2 for i, word in enumerate(list(words))} 
word2index['-PAD-'] = 0  # Valor especial usado para padding
word2index['-OOV-'] = 1  # Valor especial usado para palavras não pertencentes ao vocabulário definido em treino   
 
tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # Valor especial usado para padding

train_sentences_X, valid_sentences_X, test_sentences_X, train_tags_y, valid_tags_y, test_tags_y = [], [], [], [], [], [] #corpus que armazenam os inteiros associados
                                                                                #as palavras e tags
for sentence in train_sentences:
    sentence_int = []
    for word in sentence:
        try:
            sentence_int.append(word2index[word.lower()])
        except KeyError:
            sentence_int.append(word2index['-OOV-'])
    train_sentences_X.append(sentence_int)

for sentence in valid_sentences:
    sentence_int = []
    for word in sentence:
        try:
            sentence_int.append(word2index[word.lower()])
        except KeyError:
            sentence_int.append(word2index['-OOV-'])
    valid_sentences_X.append(sentence_int)
    
for sentence in test_sentences: 
    sentence_int = []
    for word in sentence:
        try:
            sentence_int.append(word2index[word.lower()])
        except KeyError:
            sentence_int.append(word2index['-OOV-'])
    test_sentences_X.append(sentence_int)
    
for sentence in train_tags:
    train_tags_y.append([tag2index[tag] for tag in sentence])

for sentence in valid_tags:
    valid_tags_y.append([tag2index[tag] for tag in sentence])
    
for sentence in test_tags:
    test_tags_y.append([tag2index[tag] for tag in sentence])
    
print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])

MAX_LENGTH_TRAIN = len(max(train_sentences_X, key=len))

MAX_LENGTH_TEST = len(max(test_sentences_X, key=len))

if(MAX_LENGTH_TEST > MAX_LENGTH_TRAIN):
    MAX_LENGTH = MAX_LENGTH_TEST
else: 
    MAX_LENGTH = MAX_LENGTH_TRAIN

print(MAX_LENGTH)
 
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

print(train_sentences_X[0])
print(test_sentences_X[0])
print(valid_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])
print(valid_tags_y[0])

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
 
model.summary()

categories = len(tag2index)

categorical_sequences = []
for sentence in train_tags_y:
    categ_sent = []
    for item in sentence:
        x = np.zeros(categories)
        x[item] = 1.0
        categ_sent.append(x)
    categorical_sequences.append(categ_sent)