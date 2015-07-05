library(twitteR)

setup_twitter_oauth("YGZrR9ouTty0cS103T2M1mqW3",
                    "HuXegJEpmGF3DmDEz2aR5WKAPVUWenE8G1X1IgaKSCNglB4bnv"
                   )
#,
#"178562590-ffGhDIBngzvMD51paPsnkFmUBArnhwgNiChONmHz",
#"93ghvvmgBvc7wrAGzwOnumxDCtO0ymQtGVKPjpacyKDPX"

tweets <- searchTwitter('#cancerresearch', n=10000)
tweets2=strip_retweets(tweets,strip_manual = TRUE, strip_mt = TRUE)

