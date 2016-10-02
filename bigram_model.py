import nltk
from nltk.model import build_vocabulary, count_ngrams, LaplaceNgramModel, LidstoneNgramModel

'''
lincoln_address_file = open('files/FirstInauguralAddress.txt')
raw_lincoln_address = lincoln_address_file.read().lower()
# lb_train_1 = raw_lincoln_address.lower().split()
lb_train_1_sents = nltk.sent_tokenize(raw_lincoln_address, language="english")
lb_train_1_words = nltk.word_tokenize(raw_lincoln_address, language='english')
'''

'''
gettysburg_address_file = open('files/Gettysburg.txt')
raw_gettysburg_address = gettysburg_address_file.read().lower()
# lb_train_2 = raw_gettysburg_address.lower().split()
lb_train_2_sents = nltk.sent_tokenize(raw_gettysburg_address, language="english")
lb_train_2_words = nltk.word_tokenize(raw_gettysburg_address, language='english')
'''

lb_train_file = open('files/LB-Train.txt')
raw_lb_train_file = lb_train_file.read().lower()
lb_train_words = nltk.word_tokenize(raw_lb_train_file, language='english')

lb_vocab = build_vocabulary(2, lb_train_words)
# lb_vocab = build_vocabulary(1, lb_train_1_words, lb_train_2_words)
# print(lb_vocab)

lb_train = []
lb_train.append(lb_train_words)
'''
lb_train.append(lb_train_1_words)
lb_train.append(lb_train_2_words)
'''
# print(lb_train)

lb_bigram_counts = count_ngrams(2, lb_vocab, lb_train)
# print(lb_bigram_counts.ngrams[2])
# print(sorted(lb_bigram_counts.ngrams[2].conditions()))

lb = LidstoneNgramModel(0.2, lb_bigram_counts)
# print("lincoln score ", lb.score("never", ["had"]))

lincoln_address_file_2 = open('files/SecondInauguralAddress.txt')
lb_test = lincoln_address_file_2.read().lower()
lb_test_words = nltk.word_tokenize(lb_test)
print("Perplexity of LB on LB-Test = ", lb.perplexity(lb_test_words))

'''
for ngram in lb_bigram_counts.to_ngrams(lb_test_words):
    print(ngram)
'''

'''
nelson_address_file = open('files/IamPreparedToDie.txt')
raw_nelson_address = nelson_address_file.read().lower()
# mb_train_1 = raw_nelson_address.lower().split()
mb_train_1_sents = nltk.sent_tokenize(raw_nelson_address, language="english")
mb_train_1_words = nltk.word_tokenize(raw_nelson_address, language="english")

freedom_award_file = open('files/InternationalFreedomAward.txt')
raw_freedom_award = freedom_award_file.read().lower()
# mb_train_2 = raw_freedom_award.lower().split()
mb_train_2_sents = nltk.sent_tokenize(raw_freedom_award, language='english')
mb_train_2_words = nltk.word_tokenize(raw_freedom_award, language='english')
'''

mb_train_file = open('files/MB-Train.txt')
raw_mb_train_file = mb_train_file.read().lower()
mb_train_words = nltk.word_tokenize(raw_mb_train_file, language='english')

mb_vocab = build_vocabulary(2, mb_train_words)
# mb_vocab = build_vocabulary(1, mb_train_1_words, mb_train_2_words)

mb_train = []
mb_train.append(mb_train_words)
'''
mb_train.append(lb_train_1_words)
mb_train.append(lb_train_2_words)
'''

mb_bigram_counts = count_ngrams(2, mb_vocab, mb_train)

mb = LidstoneNgramModel(0.2, mb_bigram_counts)
# print("mandela score ", mb.score("the", ["and"]))

nelson_address_file_2 = open('files/AfricanNationalCongress.txt')
mb_test = nelson_address_file_2.read()
mb_test_words = nltk.word_tokenize(mb_test)
print("Perplexity of MB on MB-Test = ", mb.perplexity(mb_test_words))

# print("Perplexity of MB on LB-Test = ", mb.perplexity(lb_test_words))
# print("Perplexity of LB on MB-Test = ", lb.perplexity(mb_test_words))
print("Perplexity of LB on LB-Train = ", lb.perplexity(lb_train_words))
print("Perplexity of MB on MB-Train = ", mb.perplexity(mb_train_words))

print("Perplexity of MB on LB-Train = ", mb.perplexity(lb_train_words))
print("Perplexity of LB on MB-Train = ", lb.perplexity(mb_train_words))


