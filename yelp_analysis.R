
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
yelp_cat_ratings=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/cuisine/review_ratings_with_ref.txt",header=FALSE)
yelp_sample_reviews<-read.table("/Users/chaitanyakelkar/Documents/capstone_vm/analysis/review_sample_100000.txt",sep="\n",quote="",header=FALSE)

yelp_indian_dishes=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task3/salient.csv",header=TRUE)



cat_aggregates=aggregate(V1~V2,yelp_cat_ratings,length)

barplot(cat_aggregates$V1,names.arg=cat_aggregates$V2,col='blue',xlab='Cuisines',ylab='Frequency',main="Cuisines/Category distribution of reviews")


#yelp_reviews_lines<-readLines("/Users/chaitanyakelkar/Documents/capstone_vm/analysis/review_sample_100000.txt")
#yelp_review_frame<-lapply(yelp_reviews_lines, function(x)as.data.frame(t(x)))
#yelp_review_frame<-NULL

cuisine_matrix_frame=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task2/2-2.similarity/cuisine_sim_matrix.csv",header=FALSE)
cuisine_labels=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task2/2-2.similarity/cuisine_indices.txt",header=FALSE)
cuisine_matrix=data.matrix(cuisine_matrix_frame, rownames.force = NA)
rownames(cuisine_matrix)=cuisine_labels[,1]
colnames(cuisine_matrix)=cuisine_labels[,1]
rownames(cuisine_matrix_frame)=cuisine_labels[,1]
colnames(cuisine_matrix_frame)=cuisine_labels[,1]
cuisine_matrix_melt=melt(cuisine_matrix)
cuisine_matrix_melt=cbind(cuisine_matrix_melt,cluster=rep(c(1,2,3),each=10))

cuisine_matrix_21_frame=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task2/2-1/cuisine_sim_matrix.csv",header=FALSE)
cuisine_labels_21=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task2/2-1/cuisine_indices.txt",header=FALSE)
cuisine_matrix_21=data.matrix(cuisine_matrix_21_frame, rownames.force = NA)
rownames(cuisine_matrix_21)=cuisine_labels_21[,1]
colnames(cuisine_matrix_21)=cuisine_labels_21[,1]

cuisine_matrix_22_frame=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task2/2-2/cuisine_sim_matrix.csv",header=FALSE)
cuisine_labels_22=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task2/2-2/cuisine_indices.txt",header=FALSE)
cuisine_matrix_22=data.matrix(cuisine_matrix_22_frame, rownames.force = NA)
rownames(cuisine_matrix_22)=cuisine_labels_22[,1]
colnames(cuisine_matrix_22)=cuisine_labels_22[,1]
cuisine_matrix_22_melt=melt(cuisine_matrix_22)
cuisine_matrix_22_melt=cbind(cuisine_matrix_22_melt,cluster=rep(c(1,2,3),each=10))

cuisine_matrix_221_frame=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task2/2-2.sim_vary/cuisine_sim_matrix.csv",header=FALSE)
cuisine_labels_221=read.csv("/Users/chaitanyakelkar/Documents/capstone_vm/task2/2-2.sim_vary/cuisine_indices.txt",header=FALSE)
cuisine_matrix_221=data.matrix(cuisine_matrix_221_frame, rownames.force = NA)
rownames(cuisine_matrix_221)=cuisine_labels_221[,1]
colnames(cuisine_matrix_221)=cuisine_labels_221[,1]
cuisine_matrix_221_melt=melt(cuisine_matrix_221)
cuisine_matrix_221_melt=cbind(cuisine_matrix_221_melt,cluster=rep(c(1,2,3),each=10))

ggplot(cuisine_matrix_melt,aes(Var1,Var2,alpha=value,fill=cluster)) +
  xlab("cuisines") +
  ylab("cuisines") +
  scale_alpha_continuous(range = c(0, 1)) +
  ggtitle("Cuisine Map") +
  geom_raster()+ # Draws the ggplot cuisine map
  scale_fill_gradientn(name="Colors just for better visuals",colours = rainbow(3)) + # Decides the colors for clusters
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1,family="Courier",size="12",face="bold")) + # Decides the text label orientation on axes (45 degrees)
  theme(axis.text.y = element_text(angle = 0, hjust = 1,family="Courier",size="12",face="bold")) + # Decides the text label orientation on axes (45 degrees)  
  geom_tile(colour = "white",size=0.5)
  #scale_x_discrete(limits=cuisine_labels_sorted_22) + # Decides the order (sorted by cluster IDs)
  #scale_y_discrete(limits=cuisine_labels_sorted_22) # Decides the order (sorted by cluster IDs)

### EXPERIMENTING K-means
k <- kmeans(cuisine_matrix_frame,5)

dfc <- cbind(cuisine_matrix_frame,id=seq(nrow(cuisine_matrix_frame)),cluster=k$cluster)
dfc <- cbind(dfc,idsort=dfc[order(dfc[,"cluster"]),"id"])

dfm<-melt(dfc,id.var=c("id","cluster"))

cuisine_melt$cluster=dfm$cluster

#-----------------------------------

## EXPERIMENTING K-Medoids,PAM
#-------------------------------

#Relative Measure
silhouette_mean=summary(silhouette(cluster_K_medoids))

distMat <- as.matrix( distancematrix(cuisine_matrix, d = "cosangle", na.rm = TRUE) )

sihouette_k_vector=lapply(2:29,function(x) { pam(distMat,x,diss=TRUE,keep.diss=TRUE) })
                                       #t<-pam(distMat,x,diss=TRUE,keep.diss=TRUE)
                                       #summary(silhouette(t))[[1]][[4]] })

sihouette_values=lapply(sihouette_k_vector,function(x) x$silinfo$avg.width)
silhouette_plot<-data.frame(unlist(sihouette_values))
silhouette_plot<-cbind(silhouette_plot,id=2:29)
colnames(silhouette_plot)=c("avg_silhouette_coefficent","id")
plot(silhouette_plot$id,silhouette_plot$avg_silhouette_coefficent,type="l",
     xlab="Number of Clusters",ylab="average silhouette coefficent",
     main="Silhouette Co-efficient Values using different number of clusters in Partitioning Around Medians (PAM)")

#Hierarchical Silhouette Plot
hc<-hclust(as.dist(distMat))
sihouette_hc_vector=lapply(2:29,function(x) { cutree(hc,x) })
sihouette_hc_values=lapply(sihouette_hc_vector,function(x) summary(silhouette(x,as.dist(distMat)))$avg.width)
silhouette_hc_plot<-data.frame(unlist(sihouette_hc_values))
silhouette_hc_plot<-cbind(silhouette_hc_plot,id=2:29)
colnames(silhouette_hc_plot)=c("avg_silhouette_coefficent","id")
plot(silhouette_hc_plot$id,silhouette_hc_plot$avg_silhouette_coefficent,type="l",
     xlab="Number of Clusters",ylab="average silhouette coefficent",
     main="Silhouette Co-efficient Values using different cutree K values from hierarchical clustering")

library(cluster)


cluster_K_medoids <- pam(distMat, k = 15, diss = TRUE, keep.diss = TRUE) #Select Best K based on average silhouette values plot
cuisine_k_medoids <- cbind(cuisine_matrix_frame,cluster=cluster_K_medoids$clustering)                                       

#cuisine_k_medoids_melt <- melt(cuisine_k_medoids,id.var=c("cluster"))
cuisine_k_medoids_melt <- data.frame(cuisine1=rep(names(cuisine_k_medoids[,-31]),each=nrow(cuisine_k_medoids)),
                                     cuisine2=rep(names(cuisine_k_medoids[,-31]),each=1,times=nrow(cuisine_k_medoids)),
                                     value=unlist(unname(cuisine_k_medoids[,-31])),
                                     cluster=rep(unlist(unname(cuisine_k_medoids[,31])),each=nrow(cuisine_k_medoids)))
#cuisine_k_medoids_melt$value <- round(cuisine_k_medoids_melt$value,digits=1)

cuisine_labels_sorted<-cuisine_labels[order(cuisine_k_medoids$cluster),]


ggplot(cuisine_k_medoids_melt,aes(cuisine1,cuisine2,alpha=value,fill=cluster)) +
  xlab("cuisines") +
  ylab("cuisines") +
  scale_alpha(range=c(0,1)) +
  ggtitle("Cuisine Map") +
  geom_raster()+ # Draws the ggplot cuisine map
  scale_fill_gradientn(name="Cluster",colours = rainbow(15)) +
  #scale_fill_gradientn(name="Cluster",colours = c("green","red","blue","orange","lightgreen","pink","violet","yellow",
  #                                                "skyblue","aquamarine","antiquewhite","magenta","cyan","darkgoldenrod","darkgreen")) + # Decides the colors for clusters
  theme_classic() +  
  theme(axis.text.x = element_text(angle = 45, hjust = 1,family="Courier",size="12",face="bold")) + # Decides the text label orientation on axes (45 degrees)
  theme(axis.text.y = element_text(angle = 0, hjust = 1,family="Courier",size="12",face="bold")) + # Decides the text label orientation on axes (45 degrees)
  scale_x_discrete(limits=cuisine_labels_sorted) + # Decides the order (sorted by cluster IDs)
  scale_y_discrete(limits=cuisine_labels_sorted) + # Decides the order (sorted by cluster IDs) +
  geom_tile(colour = "white",size=0.5)

#scale_fill_gradientn(name="Cluster",colours = c("black","red","blue","orange","green","pink","violet","yellow",
#"skyblue","aquamarine","antiquewhite","magenta","cyan","darkgoldenrod","darkgreen")) + # Decides the colors for clusters

#------------------------------------------------------------------------------------------------
#Hierarchical Clustering

hc<-hclust(as.dist(distMat))

#Silhouette Plot for hierarchical clustering
plot(silhouette(cutree(hc,15),as.dist(distMat)))

cuisine_hc_clusters <- cbind(cuisine_matrix_frame,cluster=cutree(hc,12))

cuisine_hc_clusters_melt <- data.frame(cuisine1=rep(names(cuisine_hc_clusters[,-31]),each=nrow(cuisine_hc_clusters)),
                                       cuisine2=rep(names(cuisine_hc_clusters[,-31]),each=1,times=nrow(cuisine_hc_clusters)),
                                       value=unlist(unname(cuisine_hc_clusters[,-31])),
                                       cluster=rep(unlist(unname(cuisine_hc_clusters[,31])),each=nrow(cuisine_hc_clusters)))


cuisine_labels_hc_sorted<-cuisine_labels[order(cuisine_hc_clusters$cluster),]

ggplot(cuisine_hc_clusters_melt,aes(cuisine1,cuisine2,alpha=value,fill=cluster)) +
  xlab("cuisines") +
  ylab("cuisines") +
  scale_alpha_continuous(range = c(0, 1)) +
  ggtitle("Cuisine Map") +
  geom_raster(color="black")+ # Draws the ggplot cuisine map
  scale_fill_gradientn(name="Cluster",colours = rainbow(12)) +
  #scale_fill_gradientn(name="Cluster",colours = c("green","red","blue","orange","darkcyan","pink","violet","yellow",
  #"skyblue","aquamarine","antiquewhite","magenta","cyan","darkgoldenrod","darkgreen")) + # Decides the colors for clusters
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1,family="Courier",size="12",face="bold")) + # Decides the text label orientation on axes (45 degrees)
  theme(axis.text.y = element_text(angle = 0, hjust = 1,family="Courier",size="12",face="bold")) + # Decides the text label orientation on axes (45 degrees)
  scale_x_discrete(limits=cuisine_labels_hc_sorted) + # Decides the order (sorted by cluster IDs)
  scale_y_discrete(limits=cuisine_labels_hc_sorted) + # Decides the order (sorted by cluster IDs)
  geom_tile(colour = "white",size=0.5)


#---------------------------------------------------------------------------------------------
#Dont need this
cuisine_melt_sorted<-cuisine_melt[order(cuisine_melt$cluster),]



ggplot(cuisine_melt,aes(Var1,Var2,alpha=value,fill=cluster)) + geom_raster()+ # Draws the ggplot cuisine map
  scale_fill_gradientn(name="Cluster",colours = rainbow(8) + # Decides the colors for clusters
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + # Decides the text label orientation on axes (45 degrees)
  theme(axis.text.y = element_text(angle = 45, hjust = 1)) + # Decides the text label orientation on axes (45 degrees)
  scale_x_discrete(limits=cuisine_labels_sorted) + # Decides the order (sorted by cluster IDs)
  scale_y_discrete(limits=cuisine_labels_sorted) # Decides the order (sorted by cluster IDs)
  

ggplot(melt(cuisine_matrix),aes(Var1,Var2,fill=value)) + geom_raster()

layout(cuisine_matrix)
image(as.matrix(cuisine_matrix), main = "Original Weights",col=gray((100:0)/100),xlab=cuisine_labels[,1],ylab=cuisine_labels[,1])

my_palette <- colorRampPalette(c("white","black"))(n = 50)
heatmap(cuisine_matrix_21,Rowv=NA,Colv=NA,col=my_palette)

