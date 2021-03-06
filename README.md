# Thesis-Project
MA Cog Sem Research Project - Luca Nannini 
*"Analyzing semantic contagion of mass entrainment in tweets produced during 2016 U.S. first presidential debate"*

:warning: NB. In case the .ipynb files will be not loaded correctly (error message: "Sorry, something went wrong. Reload?") try to visualize them using nbviewer online: https://nbviewer.jupyter.org/

# Data Analysis
## Data Scraping & Preprocessing

### Official Transcripts:
* Raw .csv file of the debate cycle 
  * Chronological annotations of single turns
  * Remove moderators and public speeches
  * Bag-of-Words: preliminary analysis
  * Text cleaning and word frequencies with NLTK 

### Tweets Corpus of the First Debate:
* Raw .csv file of Tweets produced during the the first debate
  * Dataframe’s features extraction 
  * Retweets removal, text cleaning and word frequencies

## Data Wrangling

### Analysis of Lexical Intersection of the First Debate:
* Syntactic overlap 
* Semantic overlap 

### Segments and Overall Topic Analysis:
* Methodologies
  * Tweet Segments
  * Debate Segments
* Segments topic analysis
  * I° segment, "Achieving Prosperity"
  * II° segment, "Candidate Figure Issues”
  * III° segment, "America's Direction"
  * IV° segment, "Securing America"
  * V° segment, "Mutual & Election Acceptance"
* Overall topic analysis visualization
  * LDA analysis
  * Histogram of mass scale attention in tweet volume
  * FastText word embeddings
  * LDA heatmap 
  * LDAvis 
  * LDA wordclouds
  * LDA word weight barcharts 
