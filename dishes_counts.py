from csv import reader



indian_dishes=[]
with open ("Indian_dishes.txt", 'r') as f:
        for line in f.readlines():
            print line
            indian_dishes.append(line.lower().strip())

indian_dishes=set(indian_dishes)

indian_dishes_reviews_counts={}
indian_dishes_restaurant_ids={}
indian_dish_postings={}
indian_dish_reviews_ratings={}
indian_dish_ratings_postings={}

with open("Indianreviews_with_ref.txt") as f:
    for line in reader(f):
        print line[2]
        for x in indian_dishes:
            if ((line[2].lower()).find(x) >= 0):
                print "found match for:"+x
                if x in indian_dishes_reviews_counts:
                    indian_dishes_reviews_counts[ x ]+=1
                    indian_dishes_restaurant_ids [ x ].append(line[0])
                    indian_dish_reviews_ratings [ x ].append(int(line[1]))
                    dish_posting_dict=indian_dish_postings [ x ]
                    if line[0] in dish_posting_dict:
                        dish_posting_dict[line[0]]+=1
                    else:
                        dish_posting_dict[line[0]]=1
                    indian_dish_postings[ x ]=dish_posting_dict
                    dish_ratings_posting_dict=indian_dish_ratings_postings [ x ]
                    if line[0] in dish_ratings_posting_dict:
                        dish_ratings_posting_dict[line[0]]+=int(line[1])
                    else:
                        dish_ratings_posting_dict[line[0]]=int(line[1])
                    indian_dish_ratings_postings[ x ]=dish_ratings_posting_dict
                else:
                    indian_dishes_reviews_counts[ x ]=1 #total reviews
                    indian_dishes_restaurant_ids [ x ]=[ line[0] ] #total restaurants
                    indian_dish_reviews_ratings [ x ] = [ int(line[1]) ] # review ratings
                    indian_dish_postings[ x ] = {line[0]:1} #total reviews by restaurant 
                    indian_dish_ratings_postings [ x ] = {line[0]:int(line[1])} #ratings by restaurant
                    

indian_dishes_reviews_counts_text=[]
indian_dishes_postings_text=[]

#Inverted Index

for dish in indian_dishes_reviews_counts:    
    total_unique_restaurants=len(set(indian_dishes_restaurant_ids [ dish ]))
    indian_dishes_reviews_counts_text.append(dish+","
                                             +str(total_unique_restaurants)+","
                                             +str(indian_dishes_reviews_counts[dish])+","
                                             +str(round(float(sum(indian_dish_reviews_ratings[dish]))/
                                                        indian_dishes_reviews_counts[dish],2)))
    dish_ratings_postings_dict=indian_dish_ratings_postings[dish]
    for rest_id,freq in indian_dish_postings[dish].items():
        average_rating=round(float(dish_ratings_postings_dict[rest_id])/freq,2)
        indian_dishes_postings_text.append(dish+","+
                                           rest_id+","+
                                           str(freq)+","+
                                           str(average_rating))


with open('indian_dishes_reviews_counts.txt','w') as f:
    f.write(u'\n'.join(indian_dishes_reviews_counts_text).encode('utf-8').strip())

with open('indian_dishes_postings.txt','w') as f:
    f.write(u'\n'.join(indian_dishes_postings_text).encode('utf-8').strip())    






    
