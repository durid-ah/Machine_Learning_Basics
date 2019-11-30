import fasttext
import re

# This script is dependent on the review_model_ngrams.bin file that can be downloaded from
# https://drive.google.com/open?id=1bA0UfHDmA5T_LpiRsPfBHJVvuzXLnNbL
# Warning! It's over 1GB in size
# Binary File: It has been trained on the yelp reviews available at:
# https://www.yelp.com/dataset/download
# the data has been preprocessed using preprocess_dataset.py
# Script Desc: Predicts the number of stars of a review


def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string


# Input reviews to be checked by the machine
reviews = [
    "This restaurant literally changed my life. This is the best food I've ever eaten!",
    "I hate this place so much. They were mean to me.",
    "I don't know. It was ok, I guess. Not really sure what to say."
]

# Pre-process the text of each review so it matches the training format
preprocessed_reviews = list(map(strip_formatting, reviews))

# Load the model
classifier = fasttext.load_model('reviews_model_ngrams.bin')

# Get fastText to classify each review with the model
labels, probabilities = classifier.predict(preprocessed_reviews, 1)

# Print the results
for review, label, probability in zip(reviews, labels, probabilities):
    stars = int(label[0][-3])

    print("* " * stars) # Print out the number of stars
    print("({}% confidence)".format(int(probability[0] * 100))) # print out the confidence of the prediction
    print(review) # Print the actual review
    print()
