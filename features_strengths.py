import numpy
numpy.corrcoef(map(int,test_cuisine_features[1][:545]), map(int,labels[:545]))[0, 1]

cuisines_positive_corr={}
for i in range(99):
    pearsons_coeff=numpy.corrcoef(map(int,test_cuisine_features[i][:546]), map(int,labels[:546]))[0, 1]
    if pearsons_coeff > 0:
        print vec.get_feature_names()[i]
        cuisines_positive_corr[vec.get_feature_names()[i]]=1
 
test_cuisine_features['labels']=labels       
        
cuisines_positive_corr={}
test_cuisine_features_subset=test_cuisine_features[test_cuisine_features['labels'] == '1']
for i in range(total_cuisine_features):
    if (sum(test_cuisine_features_subset[i][:test_cuisine_features_subset.shape[1]]) > 0):
        print vec.get_feature_names()[i]
        cuisines_positive_corr[vec.get_feature_names()[i]]=1
        

zip_codes_subset=text_reference[text_reference['labels'] == '1']
        
