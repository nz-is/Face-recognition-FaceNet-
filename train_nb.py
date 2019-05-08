from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pickle
import argparse
from sklearn.svm import SVC# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args =ap.parse_args()

print("[INFO] loading face embeddings...")
data = pickle.loads(open(args.embeddings, 'rb').read())
print(data)
print("[INFO] Encoding labels")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

#Initialize Gaussian naive_bayes classifier
gnb = SVC(C=20.0, kernel="linear", probability=True)
gnb.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args.recognizer, "wb")
f.write(pickle.dumps(gnb))
f.close()

# write the label encoder to disk
f = open(args.le, "wb")
f.write(pickle.dumps(le))
f.close()
