import logging
import glob
import argparse
from gensim import models
from gensim import matutils
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
import operator
import scipy
from sklearn import linear_model



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
    text_reference = pd.read_csv('hygiene.dat.additional',names=['cuisines','zip_code','number_of_reviews','average_rating'],header=None)
    
    zip_codes_risk = pd.read_csv('zip_codes_risk.csv',header=0)
    
    zip_codes_dict=dict(zip(zip_codes_risk['zip_code'],zip_codes_risk['risk_prob']))

    
    #label_encoder=preprocessing.LabelEncoder()
    
    text_reference['labels']=labels
    
    zip_codes_subset=text_reference[text_reference['labels'] == '1']
    
    zip_codes_unique=np.unique(zip_codes_subset['zip_code'])
    
    #zip_codes_risk=[]
    #for a_zip in zip_codes_unique:
    #    zip_labels=zip_codes_subset[zip_codes_subset['zip_code'] == a_zip]
    #    zip_label_ints=map(int,zip_labels['labels'])
    #    if sum(zip_label_ints) > 40:
    #        print a_zip
    #        zip_codes_risk.append(a_zip)
            
    
    
    #transformed_zip_codes=[]
    #for a_zip in text_reference['zip_code']:
    #    if a_zip in zip_codes_risk:
    #        transformed_zip_codes.append(1)
    #    else:
    #        transformed_zip_codes.append(0)
             
    transformed_zip_codes=[]
    for a_zip in text_reference['zip_code']:
        if a_zip in zip_codes_dict:
            transformed_zip_codes.append(zip_codes_dict[a_zip])
        else:
           transformed_zip_codes.append(0.001)
             
    transformed_reviews_counts=[]
    for review_count in text_reference['number_of_reviews']:
        if review_count < 10:
            transformed_reviews_counts.append(0.6)
        elif review_count < 20:
            transformed_reviews_counts.append(0.2)
        elif review_count < 40:
            transformed_reviews_counts.append(0.15)
        else:
            transformed_reviews_counts.append(0.05)
            
            
    transformed_ratings=[]
    for a_avg_rating in text_reference['average_rating']:
        if a_avg_rating  < 3.5:
            transformed_ratings.append(0.4)
        elif a_avg_rating  < 4:
            transformed_ratings.append(0.3)
        elif a_avg_rating  < 4.5:
            transformed_ratings.append(0.2)
        else:
            transformed_ratings.append(0.1)
                    
    
    
    cuisines_dict_list=[]
    
    for cuisine_list in text_reference['cuisines']:
        a_cuisines_dict={}
        for a_cuisine in cuisine_list.replace('[','').replace(']','').strip().split(','):    
            a_cuisines_dict[a_cuisine.strip()]=1
        cuisines_dict_list.append(a_cuisines_dict)
        
    vec = DictVectorizer()
    
    test_cuisine_features=vec.fit_transform(cuisines_dict_list).toarray()
    
    test_cuisine_features=pd.DataFrame(test_cuisine_features)
    
    test_cuisine_features['labels']=labels

    total_cuisine_features=len(vec.get_feature_names())
    cuisines_positive_corr={}
    #for i in range(total_cuisine_features):
        #pearsons_coeff=np.corrcoef(map(int,test_cuisine_features[i][:546]),map(int,labels[:546]))[0, 1]
    #    pearsons_coeff=scipy.stats.pearsonr(map(int,test_cuisine_features[i][:546]),map(int,labels[:546]))
     #   if pearsons_coeff > 0.05:
      #      print vec.get_feature_names()[i]
       #     cuisines_positive_corr[vec.get_feature_names()[i]]=1
       
    cuisines_positive_corr={}
    test_cuisine_features_subset=test_cuisine_features[test_cuisine_features['labels'] == '1']
    for i in range(total_cuisine_features):
        if (sum(test_cuisine_features_subset[i][:test_cuisine_features_subset.shape[1]]) > 10) and (vec.get_feature_names != "'Restaurants'"):
            print vec.get_feature_names()[i]
            cuisines_positive_corr[vec.get_feature_names()[i]]=1
            
    if "'Restaurants'" in cuisines_positive_corr:
        print "Restaurants removed"
        del cuisines_positive_corr["'Restaurants'"]
    
    #Now consider cuisines with only positive correlation
    cuisines_dict_list=[]
    
    for cuisine_list in text_reference['cuisines']:
        a_cuisines_dict={}
        for a_cuisine in cuisine_list.replace('[','').replace(']','').strip().split(','):   
            if a_cuisine.strip() in cuisines_positive_corr:
                a_cuisines_dict[a_cuisine.strip()]=1
        cuisines_dict_list.append(a_cuisines_dict)
        
    transformed_cuisines_features=vec.fit_transform(cuisines_dict_list).toarray()
    
    #logging.basicConfig(format='%asctime)s: %(levelname)s : %(message)s',level=logging.INFO)

    #t0 = time()
    print("Extracting features from the training dataset using a sparse vectorizer")
    X = vectorizer.fit_transform(text)
    #print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    
    # mapping from feature id to acutal word
    id2words ={}
    for i,word in enumerate(vectorizer.get_feature_names()):
        id2words[i] = word

    #t0 = time()
    print("Applying topic modeling, using LDA")
    print(str(K_clusters) + " topics")
    #corpus = matutils.Sparse2Corpus(X[546:],  documents_columns=False)
    #lda = models.ldamodel.LdaModel(corpus, num_topics=K_clusters,id2word=id2words,iterations=1000)
    #additional_corpus = matutils.Sparse2Corpus(X,  documents_columns=False)
    #lda.update(additional_corpus)
    #print("done in %fs" % (time() - t0))
    #doc_topics_list = lda.get_document_topics(corpus)
    
    #doc_topics_mat=sparse.csr_matrix((13299,K_clusters))
    #for i, doc_a in enumerate(doc_topics_list):   
     #   for (my_topic_a, weight_a) in doc_a:
     #       doc_topics_mat[i,my_topic_a] = weight_a                        
        #doc_topics[i,K_clusters] =  transformed_zip_codes[i]
        #doc_topics[i,(K_clusters+1)] = transformed_reviews[i]
        #doc_topics[i,(K_clusters+2)] = transformed_ratings[i]
        
    zip_codes_sparse=sparse.csr_matrix(transformed_zip_codes).transpose()
    
    num_of_reviews_sparse=sparse.csr_matrix(transformed_reviews_counts).transpose()
    
    #avg_rating_sparse=sparse.csr_matrix(transformed_ratings).transpose()
    
    #doc_topics=sparse.hstack([X,transformed_cuisines_features,zip_codes_sparse,num_of_reviews_sparse]).todense()
    doc_topics=X
                
        
    #output_text = []
    #for i, item in enumerate(lda.show_topics(num_topics=K_clusters, num_words=num_display_words, formatted=False)):
     #   output_text.append("Topic: " + str(i))
     #   for weight,term in item:
     #       output_text.append( term + " : " + str(weight) )

    #print "writing topics to file:", outputfile
    #with open ( outputfile, 'w' ) as f:
    #    f.write('\n'.join(output_text))

    #clf = RandomForestClassifier(n_estimators=100,n_jobs=4,verbose=3)
    clf = linear_model.LogisticRegression(intercept_scaling=3)
    #start = time.time()
    train_docs_vectors=doc_topics[:546]
    train_labels=labels[:546]  
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
 
    test_docs_vectors=doc_topics[546:]

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
    
    parser.add_argument('-K', default=50, type=int,
                       help='K is the number of topics to use when running the LDA algorithm. Default 100.')
    parser.add_argument('-featureNum', default=80000, type=int,
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
    