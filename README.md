# DataAnalysis_And_OCR

## Task 1
**Methods Used:**
  Data Filtering and Preprocessing
  K-means Clustering algorithm
  Data Visualization Using Matplotlib
  
**Inferences:**
  Data filtering includes handling empty, faulty data and selective indexing
  K-means:
    1. Choosing Random K Points
    2. Calculating Eucledian Distance by each point
    3. Implementing them as clusters and finding new median
    4. Iterate till median stops changing
  Matplotlib
    *Scatter plot is used to plot the data points
    *Scatter plot used to visualize identified clusters and test any random points on clusters

**Reference Links:**
  1. https://pandas.pydata.org/docs/user_guide/index.html#user-guide
  2. https://numpy.org/doc/stable/
  3. https://matplotlib.org/stable/api/pyplot_summary.html
  4. https://youtu.be/5w5iUbTlpMQ?si=sK4P-j5niUPpFyWx

**Results:**
    Printing filtered data
    Visualising filtered data
    Visualizing Clusters in scatter plot
    

## Task 2
**Methods Used:**
  Algorithm to identify different characters(CHARACTER SEGMENTATION)
  Standardize and Label Encode data
  Neural Network model
  Sentiment Analysis(Naive Bayes Classifier)
  Stemming and Lemmatizaton
  
**Inferences:**
    The OCR model accurately recognized characters from the images, indicating effective preprocessing and model           training.
    The sentiment analysis model correctly classified the sentiment of the text using Naive Bayes approach predicted       by the OCR, showing robust text processing and sentiment categorization.
    The combined OCR and sentiment analysis pipeline efficiently converted image data into sentiment predictions,          demonstrating the versatility of merging different machine learning techniques.
    The models performed well on test images, suggesting the training data and methods used were comprehensive and         representative of the target domain.
    
**Reference Links:**
  1. https://pandas.pydata.org/docs/user_guide/index.html#user-guide
  2. https://numpy.org/doc/stable/
  3. https://matplotlib.org/stable/api/pyplot_summary.html
  4. https://youtu.be/bte8Er0QhDg?si=Eus_O6XhIDBxTfAr
  5. https://www.analyticsvidhya.com/blog/2022/03/building-naive-bayes-classifier-from-scratch-to-perform-sentiment-analysis/
  6. ChatGPT(for character segmentation idea while i couldn't recognize the purpose of gaps between characters)

**Results:**
  The OCR model successfully recognized characters from images, and the sentiment analysis model accurately           
  identified the sentiment of the text. This combined method effectively turned image data into sentiment       
   predictions,showing how well the models work together.
