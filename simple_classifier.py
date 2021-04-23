import logging
import glob
import argparse
from gensim import models
from gensim import matutils
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import time
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing


def main(K, numfeatures, sample_file, num_display_words, outputfile):
    K_clusters = K
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=numfeatures,
                                     min_df=2, stop_words='english',ngram_range=(1,2),
                                     use_idf=True)

    text = []
    with open (sample_file, 'r') as f:
        text = f.readlines()
    
    labels = []

    with open ('hygiene.dat.labels', 'r') as f:
        labels = f.readlines()
    
    labels=map(str.strip,labels)

    

    text_reference = []

    #with open ('hygiene.data.additional', 'r') as f:
    text_reference = pd.read_csv('hygiene.dat.additional')

    #logging.basicConfig(format='%asctime)s: %(levelname)s : %(message)s',level=logging.INFO)

    #t0 = time()
    print("Extracting features from the training dataset using a sparse vectorizer")
    X = vectorizer.fit_transform(text)
    #print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    
    # mapping from feature id to acutal word
    #id2words ={}
    #for i,word in enumerate(vectorizer.get_feature_names()):
    #    id2words[i] = word

    #t0 = time()
    print("Applying topic modeling, using LDA")
    print(str(K_clusters) + " topics")
    #corpus = matutils.Sparse2Corpus(X,  documents_columns=False)
    #lda = models.ldamodel.LdaModel(corpus, num_topics=K_clusters, id2word=id2words)
    #print("done in %fs" % (time() - t0))
        
    output_text = []
    #for i, item in enumerate(lda.show_topics(num_topics=K_clusters, num_words=num_display_words, formatted=False)):
     #   output_text.append("Topic: " + str(i))
     #   for weight,term in item:
     #       output_text.append( term + " : " + str(weight) )

    #print "writing topics to file:", outputfile
    #with open ( outputfile, 'w' ) as f:
    #    f.write('\n'.join(output_text))

    clf = RandomForestClassifier(n_estimators=500,n_jobs=4,verbose=3)
    #start = time.time()
    train_docs_vectors=X[:545]
    train_labels=labels[:545]  
    train_labels=map(int,train_labels)	
    #cols=train_docs_vectors.columns[1:]

    #predicted_label=['CG']
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_docs_vectors,
  			 				  train_labels,
                                                          test_size=0.3,
                                                          random_state=0)
    clf.fit(X_train,y_train)
    
    y_pred=clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)

    print(cross_validation.cross_val_score(clf, train_docs_vectors,train_labels,cv=20,scoring="f1_macro").mean())
    #print(metrics.accuracy_score(train_labels, predicted))
    
    #print(f1_score(y_test, y_pred, average='binary')) 
    #lb = preprocessing.LabelBinarizer()
    #y_test=lb.fit(y_test)
    #y_pred=lb.fit(y_pred)
    #print(cross_validation.cross_val_score(clf,y_test,y_pred,scoring="f1"))
 
    test_docs_vectors=X[546:]

    predicted_labels=(clf.predict(test_docs_vectors)).tolist()
    
    output_text=["CG"]
    for label in predicted_labels:
	output_text.append(str(label))
       

    print "writing predicted labels to file: competition.txt"
    with open ( 'competition.txt', 'w' ) as f:
	f.writelines("%s\n" % item for item in output_text)
        
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='This program takes in a file and some parameters and generates topic modeling from the file. This program assumes the file is a line corpus, e.g. list or reviews and outputs the topic with words and weights on the console.')
    
    parser.add_argument('-f', dest='path2datafile', default="hygiene.dat", 
                       help='Specifies the file which is used by to extract the topics. The default file is "review_sample_100000.txt"')
    
    parser.add_argument('-o', dest='outputfile', default="sample_topics.txt", 
                       help='Specifies the output file for the topics, The format is as a topic number followed by a list of words with corresdponding weights of the words. The default output file is "sample_topics.txt"')
    
    parser.add_argument('-K', default=100, type=int,
                       help='K is the number of topics to use when running the LDA algorithm. Default 100.')
    parser.add_argument('-featureNum', default=50000, type=int,
                       help='feature is the number of features to keep when mapping the bag-of-words to tf-idf vectors, (eg. lenght of vectors). Default featureNum=50000')
    parser.add_argument('-displayWN', default=15,type=int,
                       help='This option specifies how many words to display for each topic. Default is 15 words for each topic.')
    parser.add_argument('--logging', action='store_true',
                       help='This option allows for logging of progress.')
    
    
    args = parser.parse_args()
    #print args
    if args.logging:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print "using input file:", args.path2datafile
    main(args.K, args.featureNum, args.path2datafile, args.displayWN, args.outputfile)
    