from kafka import KafkaProducer
import tweepy
import sys
from json import dumps


def getKafkaProducer():
    _producer = None
    try:
        _producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                                  value_serializer=lambda x: dumps(x).encode('utf-8'))
    except Exception as excep:
        print("Error while connecting to kafka")
        print(str(excep))
    finally:
        return _producer


class StreamListener(tweepy.StreamListener):
    producer = getKafkaProducer()

    def on_status(self, status):
        if status.lang == 'en':
            if hasattr(status, "extended_tweet"):
                text = status.extended_tweet["full_text"]
            else:
                text = status.text
            print(text)
            self.producer.send(str(sys.argv[6]), value=text)


if __name__ == "__main__":
    argLength = len(sys.argv)
    if (argLength != 7):
        print()
        print(
            "Usage: <script_name> <consumer key> <consumer secret> <access key> <access secret> <Comma separated tweet keywords> <kafka topic name>")
        print()
        exit(1)

    consumerKey = str(sys.argv[1])
    consumerSecret = str(sys.argv[2])
    accessKey = str(sys.argv[3])
    accessSecret = str(sys.argv[4])
    tag = str(sys.argv[5])
    filterList = tag.split(',')

    auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
    auth.set_access_token(accessKey, accessSecret)
    api = tweepy.API(auth)

    streamListener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=streamListener, tweet_mode='extended')
    stream.filter(track=filterList)