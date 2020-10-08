import re
import codecs
import os
import numpy as np
from sklearn.preprocessing import normalize

def handle_sentends(tokens) :
    sent_ends = ["?", "!", ".", "।", "|"]
    n = len(tokens)
    i = 0
    while(i < n) :
        token = tokens[i]
        for end in sent_ends :
            if end in token :
                if(len(token) != 1):
                    tokens[i] = token.replace(end, "") 
                    tokens.insert(i+1, end+"_SENT")
                    n += 1
                    i += 1
                else :
                    tokens[i] = end+"_SENT"
        i += 1
    return tokens

def corpus_preprocess(sent) :
    references = "[?.,]*\[?[0१२३४५६७८९०][१२३४५६७८९०0]*\]"
    eclipse = "\.+|\*+"
    sent = re.sub(references, "", sent)
    sent = re.sub(eclipse, " ", sent)
    
    remove_double_underscore = ["WQ", "XC", "N_NST", "N_NNP", "RB", "N_NN", "QT_QTF"]
    sent = sent.replace("WQ", "_WQ")
    sent = sent.replace("XC", "_XC")
    for tag in remove_double_underscore :
        sent = sent.replace("__"+tag, "_"+tag)
        
   
    sent = sent.replace("'_RD_PUNC", " '_RD_PUNC ")
    sent = sent.replace(",_RD_PUNC", " ,_RD_PUNC ")
    sent = sent.replace("'", " '_RD_PUNC ")
    sent = sent.replace(",_RD_PUNC", ",")
    sent = sent.replace(",", "")
    sent = sent.replace(";", " ;_RD_PUNC ")
    sent = sent.replace("|_RD_PUNC", "|")
    
    symbols = ["‘‘", "’", "‘", "“", "”", "–", ")", "("]
    for sym in symbols :
        sent = sent.replace(sym, " "+sym+"_RD_SYM ")
    sent = sent.replace("-SYM", " -_RD_SYM ")
    sent = sent.replace(")_RD_SYM", " )_RD_SYM ")
    sent = sent.replace("+SYM", " +_RD_SYM ")
    
    
    sent = sent.replace("QC", "QC ")
    sent = sent.replace("V_VB", "V_VB ")
    
    sent = sent.replace("<s>P", "<s>")
    sent = sent.replace("P>s\<", "</s>")
    
    sent = sent.replace("<s>", " <s> ")
    sent = sent.replace("<s/>", "<\s>")
    sent = sent.replace("</s>", " <\s>")
    
    separate_comma = ["CC", "FW", "IN", "JJ", "N_NN", "N_NNP", "N_NNS", "PRP", "RB", "UH", "VB", "WDT"]
    for tag in separate_comma :
        sent = sent.replace(tag+',', tag+' '+' ,_RD_PUNCT ')
    if(sent[0:3] == "<s>") :
        sent = "<s> "+sent[3:]
    elif(sent[1:4] == "<s>") :
        sent = "<s> "+sent[4:]
    else :
        sent = "<s> "+sent
    if(sent[-5:-1] == "<\s>") :
        if(sent[-6] != " ") :
            sent = sent[:-5]+" <\s>"
    elif(sent[-4:] == "<\s>") :
        if(sent[-5] != " ") :
            sent = sent[:-4] +" <\s>"
    else :
        sent = sent + " <\s>"
    return sent

def get_words_and_tags(tokens) :
    words = []
    tags = []
    for sent_tokens in tokens :
        sent_words = []
        sent_tags = []
        for i in range(len(sent_tokens)) :
            token = sent_tokens[i]
            if(token.find("_") != -1) :
                pos = token.find("_")
                sent_words.append(token[:pos])
                sent_tags.append(token[pos+1:].upper())
            elif(token == "<s>"):
                sent_words.append(token)
                sent_tags.append("START")
            elif(token == "<\s>"):
                sent_words.append(token)
                sent_tags.append("END")   
            else :
                if(token[-1] == 'P') :
                    sent_words.append(token[::-1])
                    sent_tags.append("P")
                else :
                    sent_words.append(token)
                    sent_tags.append("UN")
        words.append(sent_words)
        tags.append(sent_tags)
    return words, tags

def get_ngrams(n, tokens) :
    i = 0
    ngrams = []
    while(i < len(tokens)-n+1) :
        ngrams.append(tokens[i:i+n])
        i += 1
    return ngrams

def get_freq_dict(ngrams) :
    ngram_freq = {}
    for ngram in ngrams :
        key = ' '.join([str(elem) for elem in ngram])
        if key not in ngram_freq :
            ngram_freq[key] = 1
        else :
            ngram_freq[key] += 1
    return ngram_freq

def create_tag_transition_matrix(hindi_tags_list) :
    tags = list(sorted(set(hindi_tags_list)))

    tag_bigrams = get_ngrams(2, hindi_tags_list)
    
    tags_count = {}
    for tag in hindi_tags_list :
        if(tag not in tags_count) :
            tags_count[tag] = 1
        else :
            tags_count[tag] += 1
            
    tag_bigrams_freq = get_freq_dict(tag_bigrams)
    V = len(tag_bigrams_freq)
    
    TTP = {}
    for i in range(len(tags)) :
        TTP[tags[i]] = {}
        for j in range(len(tags)) :
            tag_pair = tags[i]+" "+tags[j]
            if tag_pair in tag_bigrams_freq :
                TTP[tags[i]][tags[j]] = (tag_bigrams_freq[tags[i]+" "+tags[j]])/(tags_count[tags[i]])
            else :
                TTP[tags[i]][tags[j]] = 0.0001
       
    return TTP

def create_word_emission_prob(hindi_tags_list, hindi_words_list, k=1) :
    tags_count = {}
    word_emission = {}
    
    for tag in hindi_tags_list :
        if(tag not in tags_count) :
            tags_count[tag] = 1
        else :
            tags_count[tag] += 1
    
   
    for i in range(len(hindi_tags_list)) :
        word = hindi_words_list[i]
        tag = hindi_tags_list[i]
        if tag not in word_emission :
            word_emission[tag] = {}
            word_emission[tag][word] = 1
        else :
            if word not in word_emission[tag] :
                word_emission[tag][word] = 1
            else :
                word_emission[tag][word] += 1
        
    V = len(set(hindi_words_list))
        
    for tag in word_emission :
        for word in word_emission[tag] :
            word_emission[tag][word] = (word_emission[tag][word]+k)/(tags_count[tag]+k*V)
        
    return word_emission

def viterbi(test_tokens, corpus_words, corpus_tags, k=1) :
    
    TTP = create_tag_transition_matrix(corpus_tags)
    WEP = create_word_emission_prob(corpus_tags, corpus_words, k)
    
    tags_count = {}
    for tag in corpus_tags :
        if(tag not in tags_count) :
            tags_count[tag] = 1
        else :
            tags_count[tag] += 1
    
    test_tokens = test_tokens[1:]
    V = len(set(corpus_words))
    tags = list(sorted(set(corpus_tags)))
    
    N = len(set(corpus_tags))
    T = len(test_tokens)
    
    
    SEQSCORE = np.zeros((T, N))
    BACKPTR = np.zeros((T, N))
    
    for i in range(N) :
        tag = tags[i]
        word = test_tokens[0]
        res = TTP["START"][tag]
        
        if tag in WEP and word in WEP[tag] :
            res *= WEP[tag][word]
        else :
            res *= k/(tags_count[tag]+k*V)
        
        SEQSCORE[0][i] = res
    
    
    for t in range(1, T) :
        for i in range(N) :
            word = test_tokens[t]
            tag = tags[i]
            options = []
            for j in range(N) :
                res = SEQSCORE[t-1][j]*TTP[tags[j]][tags[i]]
                if tag in WEP and word in WEP[tag] :
                    res *= WEP[tag][word]
                else :
                    res *= k/(tags_count[tag]+k*V)
                options.append(res)
                
            SEQSCORE[t][i] = max(options)
            max_index = options.index(max(options))
            BACKPTR[t][i] = int(max_index)

    C = [0]*T
    C[-1] = int(np.argmax(SEQSCORE[T-1]))
    
    for i in range(T-2, -1, -1) :
        C[i] = int(BACKPTR[i+1][C[i+1]])
    
            
    tagged_sent = ""       
    for i in range(len(test_tokens)-1) :
        tagged_sent += (test_tokens[i]+"_"+tags[C[i]]+" ")
    tagged_sent = tagged_sent[:-1]
    
    return tagged_sent
            
def preprocess_test(sent) :
    if(sent[0:3] == "<s>") :
        sent = "<s> "+sent[3:]
    elif(sent[1:4] == "<s>") :
        sent = "<s> "+sent[4:]
    else :
        sent = "<s> "+sent
    if(sent[-5:-1] == "<\s>") :
        if(sent[-6] != " ") :
            sent = sent[:-5]+" <\s>"
    elif(sent[-4:] == "<\s>") :
        if(sent[-5] != " ") :
            sent = sent[:-4] +" <\s>"
    else :
        sent = sent + " <\s>"
    return sent

def main() :
    files = os.listdir("Labeled-Hindi-Corpus")
    hindi_tokens = []
    for file in files:
        name = file
        file = "Labeled-Hindi-Corpus/"+file
        f = codecs.open(file, "r", encoding='utf-8')
        sents = f.readlines()
        for sent in sents :
            sent = sent.strip()
            sent = corpus_preprocess(sent)
            hindi_tokens.append(handle_sentends(sent.split()))
        f.close()  
    hindi_words, hindi_tags = get_words_and_tags(hindi_tokens)
    hindi_tags_list = [item for sublist in hindi_tags for item in sublist]
    hindi_words_list = [item for sublist in hindi_words for item in sublist]

    TTP = create_tag_transition_matrix(hindi_tags_list)
    WEP = create_word_emission_prob(hindi_tags_list, hindi_words_list, k=1)
    file_TTP = "111708049_Tag_Transition.txt"
    file_WEP = "111708049_Word_Emission.txt"
    
    with open(file_WEP, "w") as f:
        for key, nested in sorted(WEP.items()):
            print(key, file=f)
            for subkey, value in sorted(nested.items()):
                print('\t{}: {}'.format(subkey, value), file=f)
            print(file=f)
        f.close()
        
    with open(file_TTP, "w") as f:
        for key, nested in sorted(TTP.items()):
            print(key, file=f)
            for subkey, value in sorted(nested.items()):
                print('\t{}: {}'.format(subkey, value), file=f)
            print(file=f)
        f.close()

    #input_file = "111708049_Assign3_Viterbi_Input.txt"
    sents = ["आज होगा जबरदस्त मुकाबला क्रिकेट का !", "सरकार हमारी जरूरतों को पूरा करे ।", "इस नीति के बारे में कुछ भी सही नहीं है ।", "आपने क्या उस जगह पर जाने की कोशिश की थी जो मैंने आपको बताया था ?", "कई दशक पहले यहां एक बड़ा बरगद का पेड़ हुआ करता था ।", "उन्होंने कहा - भारत में आपका स्वागत है !", "दिन के अंत तक हम कितनी दूरी तय कर सकते हैं ?", "राजा और रानी बहुत घमंडी थे |", "मेरे पास फिल्म को निर्देशित करने के लिए पैसे नहीं हैं |", "नदी के पास एक झोपड़ी है इसलिए हम वहां डेरा डाल सकते हैं !"]
    output_file = "111708049_Assign3_Viterbi_Output.txt"
    #fi = codecs.open(input_file, "r", encoding='utf-8')
    fo = codecs.open(output_file, "w", encoding='utf-8')
    #sents = fi.readlines()
    for sent in sents :
        sent = sent.strip()
        sent = preprocess_test(sent)
        tagged_sent = viterbi(sent.split(), hindi_words_list, hindi_tags_list)
        fo.write(tagged_sent+"\n")
    #fi.close() 
    fo.close()

if __name__ == '__main__' :
    main()
