import sys
import json
import tweepy
import string
import random
from collections import Counter

port_num = int(sys.argv[1])
output_file = sys.argv[2]  # out.csv

N = 1
S = 100
sample = []
counterDic = Counter()

with open(output_file, 'w') as f:
    f.write("")

char_set = string.ascii_letters + "1234567890 "


class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        global N
        global sample
        global counterDic

        def getEnglishOnly(hashtags):
            if len(hashtags) == 0:
                return []
            else:
                tagList = [tag['text'] for tag in hashtags]
                cleanList = list()
                for tag in tagList:
                    if all([True if x in char_set else False for x in tag]):
                        cleanList.append(tag)
                return cleanList

        hashtags = getEnglishOnly(status.entities['hashtags'])
        title = "The number of tweets with tags from the beginning: " + str(N) + "\n"

        if len(hashtags) != 0:

            if N <= S:  # store all
                counterDic.update(hashtags)
                sample.append(hashtags)

                # for tags in sample:
                #    counterDic.update(tags)

                sorted_dic = sorted(sorted([(key, value) for (key, value) in counterDic.items()], key=lambda d: d[0]),
                                    key=lambda d: d[1], reverse=True)

                result = []
                for idx, (tag, value) in enumerate(sorted_dic):
                    if idx == 0:
                        result.append((tag, value))
                        cnt = 1
                    else:
                        if value != sorted_dic[idx - 1][1]:
                            cnt += 1
                        if cnt > 3:
                            break
                        else:
                            result.append((tag, value))

                with open(output_file, 'a') as f:
                    f.write(title)
                    for (tag, value) in result:
                        line = tag + " : " + str(value) + "\n"
                        f.write(line)
                    f.write("\n")

                N = N + 1

            else:  # N>S
                # prob: true with S/N, false with (N-S)/N
                prob = random.choices([True, False], weights=[S, N - S], k=1)[0]

                if prob:
                    # random choice of element index to be removed
                    idx = random.choice(range(S))
                    sample.remove(sample[idx])
                    sample.append(hashtags)

                    counterDic = Counter()
                    for tags in sample:
                        counterDic.update(tags)

                    sorted_dic = sorted(
                        sorted([(key, value) for (key, value) in counterDic.items()], key=lambda d: d[0]),
                        key=lambda d: d[1], reverse=True)

                    result = []
                    for idx, (tag, value) in enumerate(sorted_dic):
                        if idx == 0:
                            result.append((tag, value))
                            cnt = 1
                        else:
                            if value != sorted_dic[idx - 1][1]:
                                cnt += 1
                            if cnt > 3:
                                break
                            else:
                                result.append((tag, value))

                    with open(output_file, 'a') as f:
                        f.write(title)
                        for (tag, value) in result:
                            line = tag + " : " + str(value) + "\n"
                            f.write(line)
                        f.write("\n")

                    N = N + 1

    def on_error(self, status_code):
        if status_code == 420:
            # returning False in on_error disconnects the stream
            return False

        # returning non-False reconnects the stream, with backoff.


if __name__ == '__main__':
    
    api_key = ""
    api_key_secret = ""
    access_token = ""
    access_token_secret = ""

    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    location = [-180, -90, 180, 90]
    track = ['a']

    language = ['en']

    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener, wait_on_rate_limit=True,
                             wait_on_rate_limit_notify=True)

    myStream.filter(locations=location, track=track)
