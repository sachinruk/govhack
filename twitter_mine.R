library(twitteR)
library(wordcloud)
library(tm)

setup_twitter_oauth("YGZrR9ouTty0cS103T2M1mqW3",
                    "HuXegJEpmGF3DmDEz2aR5WKAPVUWenE8G1X1IgaKSCNglB4bnv"
                   )
#,
#"178562590-ffGhDIBngzvMD51paPsnkFmUBArnhwgNiChONmHz",
#"93ghvvmgBvc7wrAGzwOnumxDCtO0ymQtGVKPjpacyKDPX"

tweets <- searchTwitter('#cancerresearch', n=10000)
tweets2=strip_retweets(tweets,strip_manual = TRUE, strip_mt = TRUE)

#save text
tweets_text <- sapply(tweets2, function(x) x$getText())
#create corpus
tweets_text_corpus <- Corpus(VectorSource(tweets_text))
tweets_text_corpus <- tm_map(tweets_text_corpus,
                              content_transformer(function(x) iconv(enc2utf8(x), sub='byte')),
                              mc.cores=1
)
tweets_text_corpus <- tm_map(tweets_text_corpus, content_transformer(tolower), mc.cores=1)
tweets_text_corpus <- tm_map(tweets_text_corpus, removePunctuation, mc.cores=1)
tweets_text_corpus <- tm_map(tweets_text_corpus, removeNumbers, mc.cores=1)
tweets_text_corpus <- tm_map(tweets_text_corpus, function(x)removeWords(x,stopwords()), mc.cores=1)
pal <- brewer.pal(6,"Dark2")
pal <- pal[-(1)]
wordcloud(tweets_text_corpus,min.freq = 3, max.words = 25,colors = pal)