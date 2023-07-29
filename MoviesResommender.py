# Impporting dependencies
import pandas as panda
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = panda.read_csv('Dataset/movies.csv')


# printing first five rows
# print(data.head())


# printing dimensions
# print(data.shape)  #(4803, 24)


features = ['genres','keywords','tagline','cast','director']


# detting the null values to smpty string
for feature in features:
  data[feature] = data[feature].fillna('')

combinedFeatures = data['genres']+' '+data['keywords']+' '+data['tagline']+' '+data['cast']+' '+data['director']



vectorizer = TfidfVectorizer()
featureVectors =vectorizer.fit_transform(combinedFeatures)

# print(featureVectors)

# Addng similarity scoe
similarity = cosine_similarity(featureVectors)

movieName = input(' Enter your preffered movie name: ')

listMovies = data['title'].tolist()


closeMatch = difflib.get_close_matches(movieName, listMovies)
# print(closeMatch)

match = closeMatch[0]
# print(match)


movieIndex = data[data.title == match]['index'].values[0]
# print(movieIndex)

similarityScore= list(enumerate(similarity[movieIndex]))
# print(similarityScore)

# print(len(similarityScore))


sortedOrder = sorted(similarityScore, key = lambda x:x[1], reverse = True) 
# print(sortedOrder)



print('\n30 Recommended movies for you:- \n')

i = 1

for movie in sortedOrder:
    index = movie[0]
    movieID = data[data.index == index]['title'].values[0]
    if i <= 30:
        print(i, '->', movieID)
        i += 1

