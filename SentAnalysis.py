from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import json
import hashlib
from kafka import KafkaConsumer
import re, string, random
from elasticsearch import Elasticsearch


def removeNoise(tweet_tokens, stop_words=()):
    cleanTokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleanTokens.append(token.lower())
    return cleanTokens


def get_all_words(cleanTokens):
    for tokens in cleanTokens:
        for token in tokens:
            yield token


def get_tweets_for_model(cleanTokens):
    for tweetTokens in cleanTokens:
        yield dict([token, True] for token in tweetTokens)


def getSentimentAnalyzer():
    stop_words = stopwords.words('english')
    posTokens = twitter_samples.tokenized('positive_tweets.json')
    negTokens = twitter_samples.tokenized('negative_tweets.json')

    posCleanedTokens = []
    negCleanedTokens = []

    for tokens in posTokens:
        posCleanedTokens.append(removeNoise(tokens, stop_words))

    for tokens in negTokens:
        negCleanedTokens.append(removeNoise(tokens, stop_words))

    allPosWords = get_all_words(posCleanedTokens)

    freqDistPos = FreqDist(allPosWords)
    print(freqDistPos.most_common(10))

    posTokensModel = get_tweets_for_model(posCleanedTokens)
    negTokensModel = get_tweets_for_model(negCleanedTokens)

    posData = [(tweet_dict, "Positive")
               for tweet_dict in posTokensModel]

    negData = [(tweet_dict, "Negative")
               for tweet_dict in negTokensModel]

    dataset = posData + negData
    random.shuffle(dataset)
    train_data = dataset
    classifier = NaiveBayesClassifier.train(train_data)
    return classifier


def addId(data):
    j = json.dumps(data).encode('ascii', 'ignore')
    data['doc_id'] = hashlib.sha224(j).hexdigest()
    return (data['doc_id'], json.dumps(data))


def doClassify(record):
    global analyzer
    tweet = record
    tokens = removeNoise(word_tokenize(tweet))
    sentiment = analyzer.classify(dict([token, True] for token in tokens))
    data = {
        "tweet": tweet,
        "sentiment": sentiment
    }
    return data



if __name__ == "__main__":
    consumer = KafkaConsumer(bootstrap_servers='localhost:9092',
                             auto_offset_reset='earliest',
                             value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    consumer.subscribe(['test-topic'])
    analyzer = getSentimentAnalyzer()
    count = 0
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    for message in consumer:
        data = doClassify(message.value)
        es.index(index='tweet', doc_type='default', id=count, body=data)
        count += 1








