
install.packages("gplots", dependencies = TRUE)
install.packages("fpc", dependencies = TRUE)
install.packages("RColorBrewer", dependencies = TRUE)
install.packages("ggplot2", dependencies = TRUE)
install.packages("reshape2", dependencies = TRUE)
install.packages("MASS", dependencies = TRUE)
install.packages("cluster", dep = TRUE)
install.packages("kknn", dep = TRUE)
install.packages("knn", dep = TRUE)
source("http://bioconductor.org/biocLite.R")
biocLite("hopach")
library(hopach)
library(kknn)
library(reshape2)
library(gplots)
library(ggplot2)
library(MASS)
#yelp_cat_ratings=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/cuisine/review_ratings_with_ref.txt",header=FALSE)
#yelp_sample_reviews<-read.table("/Users/chaitanyakelkar/Documents/capstone_vm/analysis/review_sample_100000.txt",sep="\n",quote="",header=FALSE)

#yelp_indian_dishes=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task4/topPhrases.txt",header=TRUE,sep='\t')
#-------------------Dish Recommendation---------------------------------------
yelp_indian_dishes=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task4/indian_dishes_topics/Indian_dishes.txt",header=F)
yelp_indian_reviews=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task4/Indianreviews_with_ref.txt",header=FALSE,sep=',',quote="\"")

#test_vector=lapply(yelp_indian_dishes$V1,
#                   function(x) { Reduce("+",(lapply(yelp_indian_reviews$V3,
#                                                     function(x,y) { if(grepl(x,y)) 
#                                                                      {1}                                                        
#                                                                  else {0}
#                                                                      }))) })


yelp_indian_restaurant_ratings=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task4/Indianrest_ratings.txt",header=F)
yelp_indian_dishes_inverted_index=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task4/Indian_dishes_reviews_counts.txt",header=F)
yelp_indian_dishes_postings=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task4/Indian_dishes_postings.txt",header=F)

#Review BM25
#Reviews Frequency transformation (sublinear)
yelp_indian_dishes_rankings<-yelp_indian_dishes_inverted_index
hist(yelp_indian_dishes_rankings$total_reviews_count,50,main="Histogram of reviews counts for Indian dishes",col="blue",xlab="Reviews Counts for Indian dishes")
yelp_indian_dishes_rankings$review_tf=lapply(yelp_indian_dishes_rankings$total_reviews_count,
                                            function(x) {round(log(1+x),2)})
hist(as.numeric(yelp_indian_dishes_rankings$review_tf),50,
     main="Histogram of reviews counts (log(x+1)) for Indian dishes",col="blue",xlab="Reviews Counts for Indian dishes (sublinear transformation)")

#BM25 transformation
k=10
yelp_indian_dishes_rankings$review_bm25_tf=lapply(yelp_indian_dishes_rankings$total_reviews_count,
                                                  function(x) {round(((k+1)*x/(x+k)),2)})

plot(yelp_indian_dishes_rankings$total_reviews_count,yelp_indian_dishes_rankings$review_bm25_tf)

hist(as.numeric(yelp_indian_dishes_rankings$review_bm25_tf),50,
     main="Histogram of reviews counts (after BM25) for Indian dishes",col="blue",xlab="Reviews Counts (after BM25) for Indian dishes")

average_restaurants_count=round(mean(yelp_indian_dishes_rankings$total_restaurants_count),2)
yelp_indian_dishes_rankings$restaurant_counts_idf=lapply(yelp_indian_dishes_rankings$total_restaurants_count,
                                             function(x) {round(x/average_restaurants_count,2)})

yelp_indian_dishes_rankings$dish_simple_score=as.numeric(yelp_indian_dishes_rankings$total_reviews_count)*
  as.numeric(yelp_indian_dishes_rankings$avg_rating)

yelp_indian_dishes_rankings$rating_round=round(yelp_indian_dishes_rankings$avg_rating)

yelp_indian_dishes_rankings$dish_score=as.numeric(yelp_indian_dishes_rankings$review_tf)*
                                       as.numeric(yelp_indian_dishes_rankings$avg_rating)

yelp_indian_dishes_rankings$dish_rest_score=as.numeric(yelp_indian_dishes_rankings$total_restaurants_count)*
  as.numeric(yelp_indian_dishes_rankings$avg_rating)

yelp_indian_dishes_rankings$dish_bm25_score=as.numeric(yelp_indian_dishes_rankings$review_bm25_tf)*
  as.numeric(yelp_indian_dishes_rankings$avg_rating)


#yelp_indian_dishes_rankings$simple_rank=order(yelp_indian_dishes_rankings$dish_simple_score)

yelp_indian_dishes_rankings<-yelp_indian_dishes_rankings[order(as.numeric(yelp_indian_dishes_rankings$dish_rest_score),decreasing=T),]

yelp_indian_dishes_rankings_top20<-yelp_indian_dishes_rankings[c(1:25,seq(26,200,5)),]

ggplot(yelp_indian_dishes_rankings_top20,
       aes(x=dish,y=dish_rest_score,fill=avg_rating)) +
  geom_bar(stat="identity",width=0.9) +
  xlab("dishes") +
  ylab("score") +  
  ggtitle("Dish Rankings") +
  scale_x_discrete(limits=yelp_indian_dishes_rankings_top20$dish)+  
  theme(axis.text.x=element_blank(),
        axis.title.x=element_blank(),
        axis.text.y=element_blank())+
  #scale_fill_manual(values=c("yellow","blue","green")) + #"red",
  scale_fill_gradientn(name="Average Rating",colours = c("orange","yellow","lightgreen","green"))+
  #theme(axis.text.x = element_text(angle = 45, hjust = 1,family="Courier",size="8",face="bold")) + # Decides the text label orientation on axes (45 degrees)
  #theme(axis.text.y = element_text(angle = 0, hjust = 1,family="Courier",size="8",face="bold")) +
  geom_text(aes(label=paste(dish,' - ',round(dish_rest_score))),hjust=-0.1,angle=70) +
  coord_cartesian(xlim=c(0,65),ylim=c(0, 750))




#--------------------- Restaurant Recommendation--------------------------------
colnames(yelp_indian_reviews)=c("restaurant_id","rating","review")
colnames(yelp_indian_dishes_postings)=c("dish","restaurant_id","reviews_count","avg_rating")
restaurant_reviews_count=aggregate(yelp_indian_reviews$review,list(yelp_indian_reviews$restaurant_id),length)
colnames(restaurant_reviews_count)=c("restaurant_id","reviews_count")
hist(restaurant_reviews_count$reviews_count,50,main="Histogram of reviews counts for Indian restaurants",col="blue",xlab="Reviews Counts of Indian Restaurants")
average_restaurant_reviews_count=round(mean(restaurant_reviews_count$reviews_count),2)
restaurant_reviews_count$reviews_count_normalized=restaurant_reviews_count$reviews_count/average_restaurant_reviews_count
hist(restaurant_reviews_count$reviews_count,50,main="Histogram of reviews counts for Indian restaurants",col="blue",xlab="Reviews Counts of Indian Restaurants")

dishes_restaurant_join_rank=merge(yelp_indian_dishes_postings[yelp_indian_dishes_postings$dish %in% 
                                                                c("chicken tikka masala","garlic naan","gulab jamun"),],
                                  restaurant_reviews_count[,c("restaurant_id","reviews_count_normalized")],
                                  by="restaurant_id")

dishes_restaurant_join_rank$reviews_count_tf=lapply(dishes_restaurant_join_rank$reviews_count,
                                                    function(x) {round(log(1+x),2)})

k=10
b=0.9
dishes_restaurant_join_rank$reviews_count_bm25=mapply(
                                                      function(x,y)                            
                                                        {round(((k+1)*x/
                                                                  (x+(1-b+b*y)))                                                                                                                
                                                        ,2)},
                                                      dishes_restaurant_join_rank$reviews_count, 
                                                      dishes_restaurant_join_rank$reviews_count_normalized)
                                                      

dishes_restaurant_join_rank$reviews_count_adjusted=
  mapply(
         function(x,y){
           round(x/(1-b+b*y),2)
         },
         as.numeric(dishes_restaurant_join_rank$reviews_count),
         as.numeric(dishes_restaurant_join_rank$reviews_count_normalized))

dishes_restaurant_join_rank$dish_restaurant_score=
  as.numeric(dishes_restaurant_join_rank$reviews_count_tf)*as.numeric(dishes_restaurant_join_rank$avg_rating)

dishes_restaurant_join_rank$dish_restaurant_score_adjusted=
  as.numeric(dishes_restaurant_join_rank$reviews_count_adjusted)*as.numeric(dishes_restaurant_join_rank$avg_rating)

dishes_restaurant_join_rank$dish_restaurant_score_bm25=
  as.numeric(dishes_restaurant_join_rank$reviews_count_bm25)*as.numeric(dishes_restaurant_join_rank$avg_rating)

restaurant_rankings=aggregate(dishes_restaurant_join_rank$dish_restaurant_score_adjusted,
                              list(dishes_restaurant_join_rank$restaurant_id),
                              sum)

restaurant_avg_ratings_query=aggregate(dishes_restaurant_join_rank$avg_rating,
                                      list(dishes_restaurant_join_rank$restaurant_id),
                                     mean)

colnames(restaurant_avg_ratings_query)=c("restaurant_id","avg_rating_by_query")

colnames(restaurant_rankings)=c("restaurant_id","score")

colnames(yelp_indian_restaurant_ratings)=c("restaurant_id","name","avg_rating","full_address","city","state")

restaurant_rankings_ordered=merge(restaurant_rankings,yelp_indian_restaurant_ratings,by="restaurant_id")
#restaurant_rankings_ordered=restaurant_rankings_names[order(restaurant_rankings_names$score,decreasing=T),]
restaurant_rankings_ordered=merge(restaurant_rankings_ordered,restaurant_reviews_count,by="restaurant_id")
restaurant_rankings_ordered=merge(restaurant_rankings_ordered,restaurant_avg_ratings_query,by="restaurant_id")
restaurant_rankings_ordered=restaurant_rankings_ordered[order(restaurant_rankings_ordered$score,decreasing=T),]

restaurant_rankings_ordered_top_40<-restaurant_rankings_ordered[c(1:50,seq(51,120,10)),]

ggplot(restaurant_rankings_ordered_top_40,
       aes(x=restaurant_id,y=score,fill=avg_rating_by_query)) +
  geom_bar(stat="identity",width=0.7) +
  xlab("restaurants") +
  ylab("score") +  
  ggtitle("Restaurant Rankings") +
  scale_x_discrete(limits=restaurant_rankings_ordered_top_40$restaurant_id)+  
  #scale_fill_manual(values=c("yellow","blue","green")) + #"red",
  theme(axis.text.x=element_blank(),
        axis.title.x=element_blank(),
        axis.text.y=element_blank())+
  scale_fill_gradientn(name="Average Rating",colours = c("yellow","lightgreen","green"))+
  #theme(axis.text.x = element_text(angle = 45, hjust = 1,family="Courier",size="8",face="bold")) + # Decides the text label orientation on axes (45 degrees)
  #theme(axis.text.y = element_text(angle = 0, hjust = 1,family="Courier",size="8",face="bold")) +
  geom_text(aes(label=paste(name,' - ',round(score))),hjust=-0.1,angle=65)+
  coord_cartesian(xlim=c(0,65),ylim=c(0, 200))
  #coord_flip() +
  #scale_y_reverse() +
#------------------------Sentiment analysis---------------------------------
install_github("mannau/tm.plugin.sentiment")
install.packages("tm", dep = TRUE)
library(tm)
library(tm.plugin.sentiment)
yelp_indian_reviews$sentiment_score=lapply(yelp_indian_reviews$review,
                                           function(x) {chartSentiment(metaXTS(x))})


#Will give all sentiment stats for corpus of reviews (convert reviews into vector of characters. using as.character())
meta(score(Corpus(VectorSource(reviews))))






